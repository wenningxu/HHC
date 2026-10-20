#!/usr/bin/env python3
"""Remove only the checkpoint files explicitly authorized on 2026-09-02.

The allowlists deliberately contain complete run-directory paths.  Data, logs,
evaluation outputs, VQ checkpoints, the historical V7 baseline, and the active
HHC run are outside the allowlists and therefore cannot be removed by this tool.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path("/mnt/8T-HDD/wenning/HHC")
MMM = Path("/home/wenning/projects/MMM")
MANIFEST = WORKSPACE / "artifacts/checkpoint_cleanup_2026-09-02.json"
CHECKPOINT_SUFFIXES = {".ckpt", ".pth", ".pt"}

HHC_WRONG_RUNS = (
    "2026-08-26-12-22-40_paper_mgt_8096_300k_seed123",
    "2026-08-28-15-16-26_paper_mgt_precleanup_raw12288_300k_seed123",
    "2026-08-29-13-37-15_paper_mgt_precleanup_raw12288_historical_cache_300k_seed123",
    "2026-08-29-14-22-05_mgt_legacy_topology_matched512_paper_mask_historical_cache_300k_seed123",
    "2026-08-29-15-45-12_mgt_known_good_historical_contract_v4_exact_sampler_token38_sanity_seed114514",
    "2026-08-30-16-08-07_hhc_formal_raw12288_repaired_mgt_300k_seed123_v1",
    "2026-09-01-06-18-28_hhc_original_mmm38_window_control_raw12288_300k_seed123_v1",
    "2026-09-01-08-37-39_hhc_all_valid_loss_control_raw12288_300k_seed123_v1",
    "2026-09-01-20-32-38_hhc_global_masked_loss_control_raw12288_300k_seed123_v1",
)

MMM_DIFFUSION_DIRS = (
    MMM / "output/test/2026-01-24-17-19-39_mdm_1_test_overfit4data",
    MMM / "output/test/2026-01-25-12-11-36_mdm_2_test_overfit4data",
    MMM / "output/test/2026-01-25-13-03-36_mdm_3_test_overfit4data",
    MMM / "output/test/2026-01-25-16-19-44_mdm_test",
    MMM / "output/test/2026-01-25-18-38-19_mld_test",
    MMM / "output/test/2026-01-25-20-18-26_mld_test_diffusion_forcing",
    MMM / "output/test/2026-01-27-16-54-25_mld_dance",
    MMM / "output/test/2026-01-27-16-57-01_mld_dance",
    MMM / "output/test/2026-01-27-17-06-17_mld_dance",
    MMM / "results/dfmld_2",
    MMM / "results/dfmld_2_val",
    MMM / "results/dfmld_2_val2",
    MMM / "results/dfmld_2_val_14w",
    MMM / "results/diffusion_forcing_0",
    MMM / "results/mld_contact_0",
    MMM / "results/mld_primitive_1",
    MMM / "results/mld_primitive_1_last",
    MMM / "results/mld_primitive_3",
    # archive/diffusion_mld/train_diffusion.py writes validation weights here.
    MMM / "results/training",
    MMM / "results/training_df",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def collect() -> list[dict]:
    classified = [
        *((WORKSPACE / "output/t2i_trans" / name, "hhc_known_wrong_or_diagnostic")
          for name in HHC_WRONG_RUNS),
        *((path, "original_mmm_diffusion") for path in MMM_DIFFUSION_DIRS),
    ]
    records: list[dict] = []
    for directory, reason in classified:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
                continue
            stat = path.stat()
            records.append(
                {
                    "path": str(path),
                    "directory": str(directory),
                    "reason": reason,
                    "size_bytes": stat.st_size,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "hard_links_before": stat.st_nlink,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    records = collect()
    unique = {(r["device"], r["inode"]): r["size_bytes"] for r in records}
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "scope": {
            "hhc_wrong_run_directories": list(HHC_WRONG_RUNS),
            "mmm_diffusion_directories": [str(p) for p in MMM_DIFFUSION_DIRS],
            "preserved": [
                "active HHC 2026-09-02 run",
                "historical HHC V7 baseline",
                "all VQ/VQ-VAE checkpoints",
                "all non-checkpoint files",
            ],
        },
        "file_count": len(records),
        "logical_bytes": sum(r["size_bytes"] for r in records),
        "maximum_reclaimable_bytes": sum(unique.values()),
        "files": records,
        "deleted_count": 0,
        "errors": [],
    }
    atomic_json(MANIFEST, payload)
    if args.execute:
        for record in records:
            try:
                Path(record["path"]).unlink()
                payload["deleted_count"] += 1
            except OSError as exc:
                payload["errors"].append({"path": record["path"], "error": str(exc)})
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(MANIFEST, payload)
    print(json.dumps({k: payload[k] for k in (
        "mode", "file_count", "logical_bytes", "maximum_reclaimable_bytes",
        "deleted_count", "errors")}, indent=2))
    print(MANIFEST)
    return bool(payload["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
