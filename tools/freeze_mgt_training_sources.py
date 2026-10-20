#!/usr/bin/env python
"""Freeze the exact MGT launch sources recorded by one training manifest."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCH_SOURCES = {
    'train/train_t2i_trans.py', 'options/option_transformer.py',
    'dataset/dataset_interx_TM.py', 'dataset/dataset_interx_tokenize.py',
    'models/model_tools.py', 'models/motion_transformer.py',
    'models/two_person_transformer.py', 'models/vqvae_sep3_shape.py',
    'utils/contact_map.py', 'utils/interx_shape_augmentation.py',
    'utils/eval_trans.py', 'utils/motion_masks.py',
    'utils/motion_process.py', 'utils/runtime.py',
}
OBSERVABLE_ROOT_LAUNCH_SOURCES = BASE_LAUNCH_SOURCES | {
    'dataset/dataset_interx_tokenize_historical_root.py',
    'utils/training_telemetry.py',
}
ALLOWED_LAUNCH_SOURCE_SETS = (
    BASE_LAUNCH_SOURCES,
    OBSERVABLE_ROOT_LAUNCH_SOURCES,
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze_sources(training_manifest, output_dir, output_manifest):
    training_manifest = Path(training_manifest).resolve()
    training = json.loads(training_manifest.read_text(encoding="utf-8"))
    expected = {
        name: digest
        for name, digest in training.get("sha256", {}).items()
        if name.endswith(".py") and "/" in name
    }
    if set(expected) not in ALLOWED_LAUNCH_SOURCE_SETS:
        raise ValueError(
            "training manifest launch-source set is not a recognized immutable "
            f"contract: got={sorted(expected)}"
        )
    output_dir = Path(output_dir).resolve()
    output_manifest = Path(output_manifest).resolve()
    if output_manifest.is_file():
        existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        existing_sources = existing.get("sources", {})
        existing_valid = bool(
            existing.get("schema") == "hhc_mgt_launch_source_snapshot_v1"
            and existing.get("status") == "pass"
            and existing.get("source_count") == len(expected)
            and existing.get("training_manifest", {}).get("sha256")
            == sha256_file(training_manifest)
            and set(existing_sources) == set(expected)
        )
        if existing_valid:
            for relative, expected_hash in expected.items():
                record = existing_sources[relative]
                path = Path(record.get("path", ""))
                if (
                    not path.is_file()
                    or not path.is_relative_to(output_dir)
                    or path.stat().st_size != record.get("bytes")
                    or sha256_file(path) != expected_hash
                    or record.get("sha256") != expected_hash
                ):
                    existing_valid = False
                    break
        if existing_valid:
            return existing
    records = {}
    for relative, expected_hash in sorted(expected.items()):
        source = (ROOT / relative).resolve()
        if not source.is_file() or not source.is_relative_to(ROOT):
            raise FileNotFoundError(f"invalid launch source: {relative}")
        actual_hash = sha256_file(source)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"live source differs from training launch hash: {relative}: "
                f"{actual_hash} != {expected_hash}"
            )
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and sha256_file(destination) == expected_hash:
            pass
        else:
            temporary = destination.with_suffix(
                destination.suffix + f".tmp.{os.getpid()}"
            )
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != expected_hash:
                raise RuntimeError(f"source snapshot copy changed: {relative}")
            os.replace(temporary, destination)
        records[relative] = artifact(destination)
    payload = {
        "schema": "hhc_mgt_launch_source_snapshot_v1",
        "status": "pass",
        "training_manifest": artifact(training_manifest),
        "source_root": str(output_dir),
        "source_count": len(records),
        "sources": records,
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(output_manifest, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze_sources(
        args.training_manifest, args.output_dir, args.output_manifest
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
