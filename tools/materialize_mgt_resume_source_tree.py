#!/usr/bin/env python
"""Materialize a small immutable source tree for an exact MGT resume."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRS = ("train", "models", "utils", "options")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def copy_python_tree(source, destination):
    records = {}
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records[relative.as_posix()] = artifact(target)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    recovery_path = args.recovery.resolve()
    recovery = json.loads(recovery_path.read_text())
    if (
        recovery.get("status") != "pass"
        or recovery.get("all_snapshot_hashes_match_training_manifest") is not True
    ):
        raise ValueError("launch-source recovery has not passed")
    output_root = args.output_root.resolve()
    output_manifest = args.output.resolve()
    if output_manifest.is_file():
        existing = json.loads(output_manifest.read_text())
        if (
            existing.get("status") == "pass"
            and existing.get("recovery", {}).get("sha256") == sha256_file(recovery_path)
        ):
            for record in existing.get("files", {}).values():
                path = Path(record["path"])
                if not path.is_file() or artifact(path) != record:
                    raise ValueError("existing resume source tree has drifted")
            print(json.dumps({"status": "pass", "reused": True,
                              "output_root": str(output_root)}, indent=2))
            return
        raise ValueError("existing resume source-tree manifest is incompatible")

    output_root.mkdir(parents=True, exist_ok=True)
    files = {}
    for package in PACKAGE_DIRS:
        files.update(copy_python_tree(ROOT / package, output_root))
    dataset_target = output_root / "dataset"
    dataset_target.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "dataset").glob("*.py")):
        target = dataset_target / source.name
        shutil.copy2(source, target)
        files[target.relative_to(output_root).as_posix()] = artifact(target)

    for relative, record in recovery["sources"].items():
        source = Path(record["path"])
        if artifact(source) != record:
            raise ValueError(f"recovered launch source drifted: {relative}")
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != record["sha256"]:
            raise ValueError(f"resume source copy changed bytes: {relative}")
        files[relative] = artifact(target)

    payload = {
        "schema": "hhc_mgt_immutable_resume_source_tree_v1",
        "status": "pass",
        "recovery": artifact(recovery_path),
        "source_root": str(output_root),
        "files": files,
        "bound_source_count": len(recovery["sources"]),
        "all_bound_hashes_match_training_manifest": True,
        "implementation": artifact(Path(__file__)),
    }
    atomic_json(output_manifest, payload)
    print(json.dumps({"status": "pass", "reused": False,
                      "files": len(files), "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
