#!/usr/bin/env python
"""Convert paired SMPL-X optimizer NPZ outputs back to official 672-D HDF5."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path).resolve()
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def scalar_string(value):
    value = np.asarray(value).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def person_rotation_6d(payload, length):
    poses = np.asarray(payload["poses"], dtype=np.float64)
    if poses.shape != (length, 55, 3) or not np.isfinite(poses).all():
        raise ValueError(f"invalid optimized pose shape/value: {poses.shape}")
    matrices = axis_angle_to_matrix(torch.from_numpy(poses))
    return matrix_to_rotation_6d(matrices).float().numpy()


def pair_to_raw(p1_payload, p2_payload, length, base=None):
    """Reconstruct `[T,56,12]` raw Inter-X features from two SMPL-X records."""
    if base is None:
        raw = np.zeros((length, 56, 12), dtype=np.float32)
    else:
        raw = np.asarray(base, dtype=np.float32).copy()
        if raw.shape != (length, 56, 12):
            raise ValueError(f"invalid base raw shape {raw.shape}")
    raw[:, :55, :6] = person_rotation_6d(p1_payload, length)
    raw[:, :55, 6:12] = person_rotation_6d(p2_payload, length)
    for person, payload in enumerate((p1_payload, p2_payload)):
        translation = np.asarray(payload["trans"], dtype=np.float32)
        if translation.shape != (length, 3) or not np.isfinite(translation).all():
            raise ValueError(
                f"person {person + 1}: invalid translation {translation.shape}"
            )
        offset = person * 6
        raw[:, 55, offset:offset + 3] = translation
        raw[:, 55, offset + 3:offset + 6] = 0.0
    if not np.isfinite(raw).all():
        raise ValueError("reconstructed raw motion contains NaN/Inf")
    return raw


def validate_source_request_binding(source, requests_path, requests_sha256):
    recorded = source.attrs.get("requests_sha256")
    if isinstance(recorded, bytes):
        recorded = recorded.decode("utf-8")
    if recorded is not None:
        if recorded != requests_sha256:
            raise ValueError("source generations belong to another request file")
        return
    recorded_path = source.attrs.get("requests")
    if isinstance(recorded_path, bytes):
        recorded_path = recorded_path.decode("utf-8")
    if not recorded_path or not Path(recorded_path).is_file():
        raise ValueError("source generations lack a verifiable request binding")
    if sha256_file(recorded_path) != requests_sha256:
        raise ValueError("source generation request path has stale contents")


def load_input_provenance(input_dir, input_kind, expected_count):
    if input_kind == "optimized":
        contract_path = input_dir / "_run_contract.json"
        completion_path = input_dir / "_completion_manifest.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completion.get("complete") is not True
            or completion.get("completed_samples") != expected_count
            or completion.get("requested_samples") != expected_count
        ):
            raise ValueError("optimizer completion manifest is incomplete")
        if completion.get("contract_sha256") != contract.get("contract_sha256"):
            raise ValueError("optimizer run/completion contracts differ")
        return {
            "run_contract": artifact(contract_path),
            "completion_manifest": artifact(completion_path),
        }
    conversion_path = input_dir / "_conversion_manifest.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    if conversion.get("status") != "complete" or conversion.get("samples") != expected_count:
        raise ValueError("round-trip conversion input is incomplete")
    return {"conversion_manifest": artifact(conversion_path)}


def validate_optimizer_sidecar(
    payload, sample_key, sample_name, repeat_id, contract_sha256,
    p1_artifact, p2_artifact,
):
    """Bind inverse conversion to the exact optimizer sample contract."""
    expected = {
        "schema": "hhc_paper_contact_optimization_sample_v1",
        "sample_key": sample_key,
        "sample_name": sample_name,
        "repeat_id": int(repeat_id),
        "run_contract_sha256": contract_sha256,
        "output_p1_sha256": p1_artifact["sha256"],
        "output_p2_sha256": p2_artifact["sha256"],
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{sample_key}: optimizer sidecar contract mismatch: {mismatches}"
        )


def copy_scalar_attributes(source, target):
    for key, value in source.attrs.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, bytes, int, float, bool)):
            target.attrs[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--source-generations", required=True, type=Path)
    parser.add_argument("--optimizer-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--input-kind", choices=("optimized", "conversion-roundtrip"),
        default="optimized",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    requests_path = args.requests.resolve()
    source_path = args.source_generations.resolve()
    input_dir = args.optimizer_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(requests_path, allow_pickle=False) as request:
        names = request["names"].astype(str)
        lengths = request["lengths"].astype(np.int64)
        betas = request["betas"].astype(np.float32).reshape(-1, 2, 10)
        genders = request["gender"].astype(np.int64).reshape(-1, 2)
    if args.limit is not None:
        names, lengths = names[:args.limit], lengths[:args.limit]
        betas, genders = betas[:args.limit], genders[:args.limit]
    count = len(names)
    requests_sha256 = sha256_file(requests_path)
    input_provenance = load_input_provenance(input_dir, args.input_kind, count)
    optimizer_contract_sha256 = None
    if args.input_kind == "optimized":
        optimizer_contract_sha256 = json.loads(
            (input_dir / "_run_contract.json").read_text(encoding="utf-8")
        )["contract_sha256"]
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() and not args.overwrite:
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError(
                f"output exists without a requested verifiable resume: {output_path}"
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_resume = {
            "schema": "hhc_optimizer_npz_to_fair_h5_v1",
            "status": "complete", "input_kind": args.input_kind,
            "samples": count, "repeat_id": args.repeat_id,
            "requests": artifact(requests_path),
            "source_generations": artifact(source_path),
            "input_provenance": input_provenance,
            "implementation": {
                "converter": artifact(Path(__file__)),
                "rotation_conversions": artifact(
                    ROOT / "utils/rotation_conversions.py"
                ),
            },
            "output": artifact(output_path),
        }
        mismatches = {
            key: {"expected": value, "actual": previous.get(key)}
            for key, value in expected_resume.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(f"inverse-conversion resume mismatch: {mismatches}")
        print(json.dumps({
            "output": str(output_path), "manifest": str(manifest_path),
            "samples": count, "status": "already complete",
        }))
        return

    temporary = Path(str(output_path) + f".tmp.{os.getpid()}")
    records = []
    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as target:
            source_names = source["names"][:count].astype(str)
            source_lengths = source["lengths"][:count].astype(np.int64)
            if not np.array_equal(source_names, names):
                raise ValueError("source generation names/order differ from requests")
            if not np.array_equal(source_lengths, lengths):
                raise ValueError("source generation lengths differ from requests")
            if source["motions"].shape[0] < count or source["motions"].shape[2:] != (150, 672):
                raise ValueError(f"unexpected source shape {source['motions'].shape}")
            if not 0 <= args.repeat_id < source["motions"].shape[1]:
                raise ValueError("repeat ID is outside the source generation file")
            validate_source_request_binding(source, requests_path, requests_sha256)
            copy_scalar_attributes(source, target)
            target.attrs.update({
                "requests": str(requests_path),
                "requests_sha256": requests_sha256,
                "source_generations": str(source_path),
                "source_generations_sha256": sha256_file(source_path),
                "optimizer_dir": str(input_dir),
                "optimizer_input_kind": args.input_kind,
                "repeat_id": args.repeat_id,
                "representation": "official_interx_raw_672d",
                "rotation_conversion": "axis-angle -> matrix -> rotation-6d",
            })
            motion_dataset = target.create_dataset(
                "motions", shape=(count, 1, 150, 672), dtype="f4",
                chunks=(1, 1, 150, 672), compression="lzf",
            )
            target.create_dataset(
                "names", data=names.astype(h5py.string_dtype("utf-8"))
            )
            target.create_dataset("lengths", data=lengths)
            target.create_dataset("completed_repeats", data=np.asarray([True]))

            for index, (name, length) in enumerate(zip(names, lengths)):
                sample_key = f"{name}_r{args.repeat_id:03d}"
                p1_path = input_dir / f"{sample_key}_pred_p1.npz"
                p2_path = input_dir / f"{sample_key}_pred_p2.npz"
                with np.load(p1_path, allow_pickle=True) as source_p1:
                    p1 = {key: source_p1[key] for key in source_p1.files}
                with np.load(p2_path, allow_pickle=True) as source_p2:
                    p2 = {key: source_p2[key] for key in source_p2.files}
                for person, payload in enumerate((p1, p2)):
                    expected_gender = "male" if genders[index, person] == 0 else "female"
                    if scalar_string(payload["gender"]) != expected_gender:
                        raise ValueError(f"{sample_key}: person {person + 1} gender changed")
                    actual_betas = np.asarray(payload["betas"], dtype=np.float32).reshape(-1)[:10]
                    if not np.allclose(actual_betas, betas[index, person], atol=1e-6, rtol=0):
                        raise ValueError(f"{sample_key}: person {person + 1} betas changed")
                source_motion = np.asarray(
                    source["motions"][index, args.repeat_id], dtype=np.float32
                ).copy()
                valid = pair_to_raw(
                    p1, p2, int(length),
                    base=source_motion[:int(length)].reshape(int(length), 56, 12),
                )
                source_motion[:int(length)] = valid.reshape(int(length), 672)
                motion_dataset[index, 0] = source_motion
                record = {
                    "index": index, "sample_name": name,
                    "sample_key": sample_key, "length": int(length),
                    "p1": artifact(p1_path), "p2": artifact(p2_path),
                }
                if args.input_kind == "optimized":
                    sidecar = input_dir / f"{sample_key}.json"
                    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
                    validate_optimizer_sidecar(
                        sidecar_payload, sample_key, name, args.repeat_id,
                        optimizer_contract_sha256, record["p1"], record["p2"],
                    )
                    record["optimizer_sidecar"] = artifact(sidecar)
                records.append(record)
            target.flush()
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest = {
        "schema": "hhc_optimizer_npz_to_fair_h5_v1",
        "status": "complete",
        "input_kind": args.input_kind,
        "samples": count,
        "repeat_id": args.repeat_id,
        "requests": artifact(requests_path),
        "source_generations": artifact(source_path),
        "input_provenance": input_provenance,
        "implementation": {
            "converter": artifact(Path(__file__)),
            "rotation_conversions": artifact(ROOT / "utils/rotation_conversions.py"),
        },
        "records": records,
        "output": artifact(output_path),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "output": str(output_path), "manifest": str(manifest_path),
        "samples": count, "status": "complete",
    }))


if __name__ == "__main__":
    main()
