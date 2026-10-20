#!/usr/bin/env python3
"""Project frozen official Inter-X GT through CoShMDM's observable 262-D space."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COSH_ROOT = Path("/mnt/8T-HDD/wenning/coshmdm-code")
sys.path.insert(0, str(COSH_ROOT))

from utils.interx_projection import project_official_motion  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", type=Path,
        default=ROOT / "output/fair_interx_requests/replication_00.npz",
    )
    parser.add_argument(
        "--ground-truth", type=Path,
        default=ROOT / "output/fair_interx_ground_truth/replication_00.h5",
    )
    parser.add_argument(
        "--skeleton-root", type=Path,
        default=ROOT / "dataset/interx/skeleton",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/coshmdm_representation_oracle/main.h5",
    )
    args = parser.parse_args()

    request = np.load(args.requests, allow_pickle=False)
    names = request["names"].astype(str)
    lengths = request["lengths"].astype(np.int64)
    request_sha256 = sha256_file(args.requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")

    with h5py.File(args.ground_truth, "r") as source:
        if not np.array_equal(source["names"][:].astype(str), names):
            raise ValueError("ground-truth names differ from frozen requests")
        if not np.array_equal(source["lengths"][:], lengths):
            raise ValueError("ground-truth lengths differ from frozen requests")
        with h5py.File(temporary, "w") as target:
            motions = target.create_dataset(
                "motions", shape=(len(names), 1, 150, 672),
                dtype=np.float32, chunks=(1, 1, 150, 672),
                compression="gzip", compression_opts=1,
            )
            target.create_dataset("names", data=np.asarray(names, dtype="S"))
            target.create_dataset("lengths", data=lengths)
            target.create_dataset("completed_repeats", data=np.asarray([True]))
            for index, (name, length) in enumerate(zip(names, lengths)):
                group = int(name[1:4])
                skeleton = np.load(args.skeleton_root / f"{group}.npy")
                raw = source["motions"][index, 0, :int(length)]
                motions[index, 0] = project_official_motion(
                    raw, skeleton, int(length)
                )
                if (index + 1) % 100 == 0 or index + 1 == len(names):
                    print({"projected": index + 1, "total": len(names)}, flush=True)
            target.attrs["source"] = (
                "diagnostic CoShMDM representation oracle: official GT -> "
                "262-D observable representation -> official 672-D"
            )
            target.attrs["diagnostic_only"] = True
            target.attrs["not_a_body_controlled_fid"] = True
            target.attrs["requests"] = str(args.requests.resolve())
            target.attrs["requests_sha256"] = request_sha256
            target.attrs["ground_truth"] = str(args.ground_truth.resolve())
            target.attrs["ground_truth_sha256"] = sha256_file(args.ground_truth)
            target.attrs["adapter"] = (
                "CoShMDM 262-D representation oracle with shape-specific "
                "all-joint root fit; identity hand rotations"
            )
            target.attrs["adapter_source_sha256"] = sha256_file(
                COSH_ROOT / "utils/interx_official_adapter.py"
            )
            target.attrs["projection_source_sha256"] = sha256_file(
                COSH_ROOT / "utils/interx_projection.py"
            )
            target.flush()
    os.replace(temporary, args.output)
    result = {
        "schema": "hhc_coshmdm_representation_oracle_export_v1",
        "status": "complete",
        "samples": len(names),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "requests_sha256": request_sha256,
        "diagnostic_only": True,
        "body_controlled_fid": False,
    }
    manifest = args.output.parent / "export_manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
