#!/usr/bin/env python3
"""Select safe projected shape-optimization outputs with fail-closed fallback.

For every completed optimizer sample, this tool constructs the preregistered
constant-delta/lower-body-preserving projection and compares it to the frozen
random-shape input.  The projection is retained only when contact mismatch
strictly improves, penetration does not increase, and foot skate is unchanged
within tolerance.  Otherwise the random-shape input is copied byte-for-byte.

The output deliberately retains the optimizer shard's original selection
positions so multiple selected shards can be passed directly to
``materialize_interx_shape_augmentation.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import PeneCalculator, npz_to_amass_dict, to_torch_params
from tools.diagnose_original_mmm_upper_body_projection import project


SCHEMA = "hhc_interx_shape_selective_projection_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def load_params(path: Path, device: torch.device) -> dict:
    with np.load(path, allow_pickle=True) as handle:
        data = {key: handle[key] for key in handle.files}
    return to_torch_params(npz_to_amass_dict(data), device)


@torch.no_grad()
def compute_metrics(
    calculator: PeneCalculator,
    p1_path: Path,
    p2_path: Path,
    contacts: list,
) -> dict[str, float]:
    """Compute all three gates from one geometry realization per person."""
    params1 = load_params(p1_path, calculator.device)
    params2 = load_params(p2_path, calculator.device)
    frames = min(params1["trans"].shape[0], params2["trans"].shape[0], 150)
    for params in (params1, params2):
        for key in ("root_orient", "pose_body", "pose_lhand", "pose_rhand", "trans"):
            params[key] = params[key][:frames]
    verts1, joints1, normals1 = calculator._compute_geometry(params1)
    verts2, joints2, normals2 = calculator._compute_geometry(params2)
    skate = calculator._calc_skate_metric(joints1) + calculator._calc_skate_metric(joints2)
    penetration = calculator._calc_penetration_metric_per_frame(
        verts1, verts2, normals1, normals2
    ).mean()
    mismatch_per_frame, has_mismatch = calculator._calc_contact_mismatch_per_frame(
        verts1, verts2, normals1, normals2, contacts
    )
    mismatch_frames = has_mismatch.sum()
    mismatch = (
        mismatch_per_frame.sum() / mismatch_frames.float()
        if int(mismatch_frames.item()) > 0
        else torch.zeros((), device=calculator.device)
    )
    return {
        "mismatch": float(mismatch.item()),
        "pene": float(penetration.item()),
        "skate": float(skate.item()),
    }


def project_atomic(initial: Path, optimized: Path, output: Path) -> None:
    temporary = output.parent / f".{output.stem}.tmp.{os.getpid()}.npz"
    project(initial, optimized, temporary, "constant_delta")
    os.replace(temporary, output)


def accept_candidate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    skate_tolerance: float,
) -> tuple[bool, dict[str, float]]:
    delta = {metric: candidate[metric] - baseline[metric] for metric in baseline}
    keep = (
        delta["mismatch"] < -1e-8
        and delta["pene"] <= 1e-8
        and abs(delta["skate"]) <= skate_tolerance
    )
    return keep, delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument(
        "--random-dir", type=Path,
        default=Path("output/interx_shape_aug_random_seed123_v2"),
    )
    parser.add_argument(
        "--contact-dir", type=Path,
        default=Path("dataset/interx/contact_pairs_codex_luna_strict_v3_train"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skate-tolerance", type=float, default=2e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimized_dir = args.optimized_dir.resolve()
    random_dir = args.random_dir.resolve()
    contact_dir = args.contact_dir.resolve()
    output_dir = args.output_dir.resolve()
    completion_path = optimized_dir / "_completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise ValueError(f"optimizer shard is incomplete: {completion_path}")
    names = completion["completed_names"]
    positions = completion.get("selection", {}).get("full_selection_positions")
    if positions is None or len(positions) != len(names):
        raise ValueError("optimizer completion has no exact full-selection positions")

    contract = {
        "schema": SCHEMA,
        "optimized_completion": str(completion_path),
        "optimized_completion_sha256": sha256_file(completion_path),
        "random_completion": str(random_dir / "_completion_manifest.json"),
        "random_completion_sha256": sha256_file(random_dir / "_completion_manifest.json"),
        "contact_manifest": str(contact_dir / "_extraction_manifest.json"),
        "contact_manifest_sha256": sha256_file(contact_dir / "_extraction_manifest.json"),
        "projection": "preserve SMPL-X joints 0,1,2,4,5,7,8,10,11 and retain only time-constant mean root-translation delta",
        "selection": "mismatch strictly improves; penetration does not increase; absolute skate delta <= tolerance; otherwise frozen random-shape fallback",
        "skate_tolerance": args.skate_tolerance,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    contract_sha256 = canonical_sha256(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_status = output_dir / "_selection_status.json"
    if existing_status.exists():
        status = json.loads(existing_status.read_text(encoding="utf-8"))
        if status.get("contract_sha256") != contract_sha256:
            raise ValueError("existing output has another immutable selection contract")
    else:
        atomic_json(existing_status, {
            "schema": SCHEMA,
            "status": "running",
            "contract": contract,
            "contract_sha256": contract_sha256,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    calculator = PeneCalculator(device=args.device)
    completed_names: list[str] = []
    accepted = 0
    deltas = {metric: [] for metric in ("mismatch", "pene", "skate")}
    for index, (name, position) in enumerate(zip(names, positions)):
        out_p1 = output_dir / f"{name}_p1.npz"
        out_p2 = output_dir / f"{name}_p2.npz"
        out_sidecar = output_dir / f"{name}.json"
        if out_p1.exists() and out_p2.exists() and out_sidecar.exists():
            metadata = json.loads(out_sidecar.read_text(encoding="utf-8"))
            if (
                metadata.get("selection_contract_sha256") != contract_sha256
                or metadata.get("output_p1_sha256") != sha256_file(out_p1)
                or metadata.get("output_p2_sha256") != sha256_file(out_p2)
            ):
                raise ValueError(f"{name}: invalid resumable selected output")
        else:
            random_paths = [random_dir / f"{name}_p{person}.npz" for person in (1, 2)]
            optimized_paths = [optimized_dir / f"{name}_p{person}.npz" for person in (1, 2)]
            for path in random_paths + optimized_paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            optimized_sidecar_path = optimized_dir / f"{name}.json"
            optimized_sidecar = json.loads(
                optimized_sidecar_path.read_text(encoding="utf-8")
            )
            source_name = optimized_sidecar["source_name"]
            with (contact_dir / f"{source_name}.pkl").open("rb") as handle:
                contacts = pickle.load(handle)
            temporary_projected = [
                output_dir / f".{name}_p{person}.projected.{os.getpid()}.npz"
                for person in (1, 2)
            ]
            for initial, optimized, projected in zip(
                random_paths, optimized_paths, temporary_projected
            ):
                project_atomic(initial, optimized, projected)
            baseline = compute_metrics(calculator, *random_paths, contacts)
            candidate = compute_metrics(calculator, *temporary_projected, contacts)
            keep, delta = accept_candidate(
                baseline, candidate, args.skate_tolerance
            )
            selected_paths = temporary_projected if keep else random_paths
            for selected, output in zip(selected_paths, (out_p1, out_p2)):
                atomic_copy(selected, output)
            for temporary in temporary_projected:
                temporary.unlink(missing_ok=True)
            metadata = {
                **optimized_sidecar,
                # Preserve the released optimizer sample schema and run-contract
                # binding required by the materialization validator. The
                # additional selection block records the postprocessing layer.
                "schema": "hhc_interx_optimized_shape_augmentation_sample_v1",
                "selection_position": int(position),
                "selection_contract_sha256": contract_sha256,
                "selective_projection": {
                    "decision": "projected_optimization" if keep else "random_shape_fallback",
                    "baseline": baseline,
                    "candidate": candidate,
                    "delta_candidate_minus_baseline": delta,
                    "skate_tolerance": args.skate_tolerance,
                    "optimized_sidecar": str(optimized_sidecar_path),
                    "optimized_sidecar_sha256": sha256_file(optimized_sidecar_path),
                    "random_p1_sha256": sha256_file(random_paths[0]),
                    "random_p2_sha256": sha256_file(random_paths[1]),
                },
                "output_p1_sha256": sha256_file(out_p1),
                "output_p2_sha256": sha256_file(out_p2),
            }
            atomic_json(out_sidecar, metadata)
        metadata = json.loads(out_sidecar.read_text(encoding="utf-8"))
        decision = metadata["selective_projection"]
        accepted += decision["decision"] == "projected_optimization"
        for metric, value in decision["delta_candidate_minus_baseline"].items():
            selected_delta = value if decision["decision"] == "projected_optimization" else 0.0
            deltas[metric].append(float(selected_delta))
        completed_names.append(name)
        if (index + 1) % 10 == 0 or index + 1 == len(names):
            atomic_json(output_dir / "_completion_manifest.json", {
                **completion,
                "complete": index + 1 == len(names),
                "completed_samples": index + 1,
                "completed_names": completed_names,
                "optimizer": {
                    **completion["optimizer"],
                    "mode": "optimized",
                    "postprocessing": "selective_constant_delta_projection_with_random_shape_fallback",
                },
                "selective_projection": {
                    "schema": SCHEMA,
                    "contract": contract,
                    "contract_sha256": contract_sha256,
                    "accepted": accepted,
                    "acceptance_rate": accepted / (index + 1),
                    "selected_mean_deltas": {
                        metric: float(np.mean(values)) for metric, values in deltas.items()
                    },
                },
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(
                json.dumps({
                    "completed": index + 1,
                    "requested": len(names),
                    "accepted": accepted,
                    "acceptance_rate": accepted / (index + 1),
                }),
                flush=True,
            )

    selected_completion = output_dir / "_completion_manifest.json"
    atomic_json(existing_status, {
        "schema": SCHEMA,
        "status": "complete",
        "contract_sha256": contract_sha256,
        "completion": str(selected_completion),
        "completion_sha256": sha256_file(selected_completion),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
