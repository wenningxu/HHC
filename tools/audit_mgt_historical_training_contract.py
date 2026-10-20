#!/usr/bin/env python
"""Freeze the authoritative 2026-02-13 low-FID MGT training contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_historical_mmm_local_history import (
    recovery_manifest_record,
    snapshot_interval,
)


RUN_DIRECTORY = Path(
    "/home/wenning/projects/MMM/wandb/"
    "run-20260213_015522-vq_mask_gen_4_ablation_adalnshape_1_cfg"
)
RECOVERED_SOURCES = {
    "train": ROOT / (
        "output/historical_mmm_local_history_train/"
        "1771033604763_set-424245_change-424244_ContentChange.py"
    ),
    "transformer": ROOT / (
        "output/historical_mmm_local_history/"
        "1771299984324_set-427147_change-427146_ContentChange.py"
    ),
    "evaluation": ROOT / (
        "output/historical_mmm_local_history_eval/"
        "1771033639074_set-424251_change-424250_ContentChange.py"
    ),
    "vq": ROOT / (
        "output/historical_mmm_local_history_vq/"
        "1772118531397_set-429297_change-429296_ContentChange.py"
    ),
}
LIVE_UNCHANGED_SOURCES = {
    "dataset_interx_TM": Path(
        "/home/wenning/projects/MMM/dataset/dataset_interx_TM.py"
    ),
    "optimizer_factory": Path(
        "/home/wenning/projects/MMM/utils/utils_model.py"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def recovered_record(path: Path, run_start: datetime) -> dict:
    interval = snapshot_interval(path)
    lower = parse_iso(interval["state_valid_from_previous_change_utc"])
    upper = parse_iso(interval["state_valid_until_snapshot_change_utc"])
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "interval": interval,
        "run_start_inside_interval": lower <= run_start < upper,
        "recovery_manifest": recovery_manifest_record(path),
    }


def live_record(path: Path, run_start: datetime) -> dict:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "modified_utc": modified.isoformat(),
        "modified_before_run_start": modified < run_start,
    }


def config_values(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        key: value.get("value") if isinstance(value, dict) else value
        for key, value in payload.items()
        if key != "_wandb"
    }


def validation_log(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"Eva\. Iter\s+(\d+)\s*:,.*?FID\.\s+([0-9]+(?:\.[0-9]+)?)",
        flags=re.DOTALL,
    )
    return [
        {"iteration": int(iteration), "fid_printed": float(fid)}
        for iteration, fid in pattern.findall(text)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=RUN_DIRECTORY)
    parser.add_argument(
        "--remote-history", type=Path,
        default=ROOT / "output/historical_mmm_wandb_remote_history.json",
    )
    parser.add_argument(
        "--checkpoint-audit", type=Path,
        default=ROOT / "output/historical_mgt_original_checkpoint_audit.json",
    )
    parser.add_argument(
        "--initialization-audit", type=Path,
        default=ROOT / "output/historical_mgt_initialization_contract_audit.json",
    )
    parser.add_argument(
        "--sampler-audit", type=Path,
        default=ROOT / "output/historical_mgt_sampler_behavior_audit.json",
    )
    parser.add_argument(
        "--loss-audit", type=Path,
        default=ROOT / "output/historical_mgt_loss_reduction_audit.json",
    )
    parser.add_argument(
        "--dependency-audit", type=Path,
        default=ROOT / "output/historical_mgt_dependency_contract_audit.json",
    )
    parser.add_argument(
        "--module-mode-audit", type=Path,
        default=ROOT / "output/historical_mgt_module_mode_contract_audit.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/historical_mgt_original_training_contract_audit.json",
    )
    args = parser.parse_args()

    files = args.run_directory / "files"
    paths = {
        "metadata": files / "wandb-metadata.json",
        "config": files / "config.yaml",
        "requirements": files / "requirements.txt",
        "summary": files / "wandb-summary.json",
        "output_log": files / "output.log",
        "transaction_log": args.run_directory / (
            "run-vq_mask_gen_4_ablation_adalnshape_1_cfg.wandb"
        ),
    }
    for label, path in {
        **paths,
        "remote_history": args.remote_history,
        "checkpoint_audit": args.checkpoint_audit,
        "initialization_audit": args.initialization_audit,
        "sampler_audit": args.sampler_audit,
        "loss_audit": args.loss_audit,
        "dependency_audit": args.dependency_audit,
        "module_mode_audit": args.module_mode_audit,
        **RECOVERED_SOURCES,
        **LIVE_UNCHANGED_SOURCES,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    remote = json.loads(args.remote_history.read_text(encoding="utf-8"))
    checkpoint = json.loads(args.checkpoint_audit.read_text(encoding="utf-8"))
    initialization = json.loads(
        args.initialization_audit.read_text(encoding="utf-8")
    )
    sampler = json.loads(args.sampler_audit.read_text(encoding="utf-8"))
    loss_audit = json.loads(args.loss_audit.read_text(encoding="utf-8"))
    dependency_audit = json.loads(
        args.dependency_audit.read_text(encoding="utf-8")
    )
    module_mode_audit = json.loads(
        args.module_mode_audit.read_text(encoding="utf-8")
    )
    config = config_values(paths["config"])
    historical_requirements = {}
    for line in paths["requirements"].read_text(encoding="utf-8").splitlines():
        if "==" in line:
            package, version = line.strip().split("==", 1)
            historical_requirements[package.lower()] = version
    environment_packages = {
        "torch": "torch",
        "lightning": "lightning",
        "pytorch-lightning": "pytorch-lightning",
        "numpy": "numpy",
        "wandb": "wandb",
    }
    current_environment = {
        label: importlib.metadata.version(package)
        for label, package in environment_packages.items()
    }
    environment_matches = {
        label: current_environment[label] == historical_requirements.get(label)
        for label in environment_packages
    }
    run_start = parse_iso(metadata["startedAt"])
    milestones = validation_log(paths["output_log"])
    expected_iterations = [0] + list(range(10_000, 300_001, 10_000))

    recovered = {
        name: recovered_record(path, run_start)
        for name, path in RECOVERED_SOURCES.items()
    }
    live = {
        name: live_record(path, run_start)
        for name, path in LIVE_UNCHANGED_SOURCES.items()
    }
    required_config = {
        "batch_size": 64,
        "total_iter": 300000,
        "warm_up_iter": 1000,
        "lr": 0.0002,
        "weight_decay": 1e-6,
        "lr_scheduler": [150000],
        "gamma": 0.05,
        "optimizer": "adamw",
        "decay_option": "all",
        "nb_code": 12288,
        "code_dim": 256,
        "motion_rep": "raw",
        "block_size": 51,
        "embed_dim_gpt": 1024,
        "n_head_gpt": 16,
        "num_layers": 9,
        "num_local_layer": 2,
        "ff_rate": 4,
        "contact": False,
        "shape": True,
        "seed": 114514,
        "pkeep": 0.5,
    }
    config_matches = {}
    for key, expected in required_config.items():
        actual = config.get(key)
        if isinstance(expected, float) and isinstance(actual, str):
            try:
                actual = float(actual)
            except ValueError:
                pass
        config_matches[key] = actual == expected
    train_source = RECOVERED_SOURCES["train"].read_text(encoding="utf-8")
    dataset_source = LIVE_UNCHANGED_SOURCES["dataset_interx_TM"].read_text(
        encoding="utf-8"
    )
    optimizer_source = LIVE_UNCHANGED_SOURCES["optimizer_factory"].read_text(
        encoding="utf-8"
    )
    remote_history = remote.get("validation_history", [])
    checks = {
        "authoritative_started_at": metadata.get("startedAt")
        == "2026-02-13T01:55:22.272525Z",
        "authoritative_git_base": metadata.get("git", {}).get("commit")
        == "8c3e7ad107ed717b878779724958cf6eff00938e",
        "all_recovered_sources_cover_run_start": all(
            record["run_start_inside_interval"] for record in recovered.values()
        ),
        "all_recovered_sources_vfs_verified": all(
            record["recovery_manifest"]["verified"]
            for record in recovered.values()
        ),
        "unchanged_live_sources_predate_run": all(
            record["modified_before_run_start"] for record in live.values()
        ),
        "all_recorded_config_matches": all(config_matches.values()),
        "current_runtime_matches_historical_requirements": all(
            environment_matches.values()
        ),
        "training_token_window_is_38": "max_motion_length=38" in dataset_source,
        "historical_optimizer_betas": "betas=(0.5, 0.9)" in optimizer_source,
        "historical_multistep_scheduler": "MultiStepLR" in train_source,
        "historical_precision_bf16": 'precision="bf16-mixed"' in train_source,
        "complete_initial_plus_30_milestones": [
            row["iteration"] for row in milestones
        ] == expected_iterations,
        "initial_fid_matches": bool(milestones)
        and milestones[0]["fid_printed"] == 25.7493,
        "ten_k_fid_matches": len(milestones) > 1
        and milestones[1]["fid_printed"] == 0.6979,
        "twenty_k_fid_matches": len(milestones) > 2
        and milestones[2]["fid_printed"] == 0.6424,
        "terminal_fid_matches": bool(milestones)
        and milestones[-1]["fid_printed"] == 0.1809,
        "summary_terminal_step": summary.get("trainer/global_step") == 299999,
        "summary_terminal_fid": summary.get("val/FID")
        == 0.18090814352035522,
        "remote_history_passes": remote.get("status") == "pass"
        and remote.get("run_state") == "finished"
        and remote.get("validation_milestones") == 30,
        "remote_first_fid_matches": bool(remote_history)
        and remote_history[0].get("val/FID") == 0.6979351043701172,
        "remote_terminal_fid_matches": bool(remote_history)
        and remote_history[-1].get("val/FID") == 0.18090814352035522,
        "checkpoint_200k_passes": checkpoint.get("status") == "pass"
        and checkpoint.get("global_step") == 200000
        and checkpoint.get("model_state", {}).get(
            "all_floating_tensors_finite"
        ) is True,
        "initialization_audit_passes": initialization.get("status") == "pass",
        "historical_sampler_behavior_passes": sampler.get("status") == "pass",
        "historical_loss_reduction_passes": loss_audit.get("status") == "pass",
        "historical_dependency_contract_passes": (
            dependency_audit.get("status") == "pass"
        ),
        "historical_module_mode_contract_passes": (
            module_mode_audit.get("status") == "pass"
        ),
        "body_controlled_fid_absent": all(
            "body_controlled" not in path.read_text(
                encoding="utf-8", errors="replace"
            ).lower()
            for path in (paths["config"], paths["summary"], paths["output_log"])
        ),
    }

    artifact_records = {
        label: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in {
            **paths,
            "remote_history": args.remote_history,
            "checkpoint_audit": args.checkpoint_audit,
            "initialization_audit": args.initialization_audit,
            "sampler_audit": args.sampler_audit,
            "loss_audit": args.loss_audit,
            "dependency_audit": args.dependency_audit,
            "module_mode_audit": args.module_mode_audit,
        }.items()
    }
    payload = {
        "schema": "historical_mgt_original_training_contract_audit_v1",
        "status": "pass" if all(checks.values()) else "mismatch",
        "role": (
            "Authoritative pre-cleanup diagnostic reference; not a formal "
            "common-50 main-table result."
        ),
        "run": {
            "directory": str(args.run_directory),
            "started_at": metadata["startedAt"],
            "finished_evidence_utc": datetime.fromtimestamp(
                paths["output_log"].stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "git": metadata.get("git"),
            "required_configuration": required_config,
            "configuration_checks": config_matches,
            "historical_runtime_requirements": {
                label: historical_requirements.get(label)
                for label in environment_packages
            },
            "current_runtime": current_environment,
            "runtime_version_checks": environment_matches,
            "summary": summary,
            "validation_milestones": milestones,
        },
        "recovered_sources": recovered,
        "unchanged_live_sources": live,
        "artifacts": artifact_records,
        "checks": checks,
        "body_controlled_fid_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "status": payload["status"],
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
