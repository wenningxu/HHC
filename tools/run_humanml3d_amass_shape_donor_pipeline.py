#!/usr/bin/env python3
"""Build and freeze the HumanML3D-train AMASS shape-augmentation plan.

The licensed AMASS, SMPL+H and model-correspondence assets are never
downloaded by this script.  Once they are present, this command runs the
hash-bound inventory, official SMPL+H-to-SMPL-X transfer, transfer quality
gate, and deterministic Inter-X augmentation assignment as one resumable
pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("audit", "inventory", "prepare", "transfer", "collect", "plan")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("RUN", json.dumps(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fitted_gender_complete(transfer_manifest: dict, gender: str) -> bool:
    expected = {
        record["identity_id"]
        for record in transfer_manifest["mesh_records"]
        if record["gender"] == gender
    }
    fitted_dir = Path(transfer_manifest["runtime"]["working_directory"])
    # The fitted directories are siblings of the frozen input manifest.
    input_manifest = Path(transfer_manifest["_manifest_path"])
    fitted_dir = input_manifest.parent / "fitted" / gender
    present = {path.stem for path in fitted_dir.glob("*.pkl")}
    return bool(expected) and expected <= present


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--humanml3d-root", type=Path, default=Path("/mnt/8T-HDD/HumanML3D")
    )
    parser.add_argument(
        "--amass-root",
        type=Path,
        default=Path("/mnt/8T-HDD/HumanML3D/amass_data"),
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=Path("/mnt/8T-HDD/wenning/smplx-transfer"),
    )
    parser.add_argument(
        "--trust-ncg-root",
        type=Path,
        default=Path("/mnt/8T-HDD/wenning/torch-trust-ncg"),
    )
    parser.add_argument(
        "--smplx-model-root",
        type=Path,
        default=Path("/home/wenning/projects/MMM/body_models"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/humanml3d_amass_shape_pipeline_v1"),
    )
    parser.add_argument(
        "--plan-output-dir",
        type=Path,
        default=Path(
            "configs/interx_shape_augmentation/"
            "humanml3d_train_amass_seed123_v1"
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--maximum-rmse-m", type=float, default=0.01)
    parser.add_argument("--stop-after", choices=STAGES, default="plan")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip a gender transfer only when all frozen identity PKLs exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    audit_path = output_root / "asset_audit.json"
    run([
        python, "tools/audit_humanml3d_amass_shape_assets.py",
        "--humanml3d-root", str(args.humanml3d_root.resolve()),
        "--amass-root", str(args.amass_root.resolve()),
        "--transfer-root", str(args.transfer_root.resolve()),
        "--smplx-model-root", str(args.smplx_model_root.resolve()),
        "--output", str(audit_path),
    ])
    audit = load_json(audit_path)
    if audit.get("status") != "pass":
        missing_models = [
            name for name, record in
            audit["required_model_and_transfer_assets"].items()
            if not record["present"]
        ]
        raise SystemExit(
            "licensed source assets are incomplete: "
            f"AMASS {audit['amass']['present_sequences']}/"
            f"{audit['humanml3d']['unique_amass_source_sequences_required']}; "
            f"missing model/transfer assets={missing_models}; audit={audit_path}"
        )
    if args.stop_after == "audit":
        return

    inventory_dir = output_root / "inventory"
    run([
        python, "tools/build_humanml3d_amass_shape_inventory.py",
        "--humanml3d-root", str(args.humanml3d_root.resolve()),
        "--amass-root", str(args.amass_root.resolve()),
        "--output-dir", str(inventory_dir),
    ])
    inventory_manifest = inventory_dir / "smplh_shape_inventory_manifest.json"
    if args.stop_after == "inventory":
        return

    transfer_dir = output_root / "transfer"
    prepare_command = [
        python, "tools/prepare_humanml3d_amass_shape_transfer.py",
        "--inventory-manifest", str(inventory_manifest),
        "--humanml3d-root", str(args.humanml3d_root.resolve()),
        "--human-body-prior-root", str(args.humanml3d_root.resolve()),
        "--smplx-model-root", str(args.smplx_model_root.resolve()),
        "--transfer-root", str(args.transfer_root.resolve()),
        "--trust-ncg-root", str(args.trust_ncg_root.resolve()),
        "--output-dir", str(transfer_dir),
    ]
    if args.resume:
        prepare_command.append("--resume")
    run(prepare_command)
    transfer_manifest_path = transfer_dir / "transfer_input_manifest.json"
    if args.stop_after == "prepare":
        return

    transfer_manifest = load_json(transfer_manifest_path)
    transfer_manifest["_manifest_path"] = str(transfer_manifest_path.resolve())
    for gender in ("female", "male"):
        if args.resume and fitted_gender_complete(transfer_manifest, gender):
            print(f"SKIP complete official transfer gender={gender}", flush=True)
            continue
        run(
            [str(value) for value in transfer_manifest["commands"][gender]],
            cwd=Path(transfer_manifest["runtime"]["working_directory"]),
        )
        if not fitted_gender_complete(transfer_manifest, gender):
            raise RuntimeError(f"official transfer did not complete gender={gender}")
    if args.stop_after == "transfer":
        return

    donor_dir = output_root / "donors"
    run([
        python, "tools/collect_humanml3d_amass_smplx_donors.py",
        "--transfer-input-manifest", str(transfer_manifest_path),
        "--maximum-rmse-m", str(args.maximum_rmse_m),
        "--output-dir", str(donor_dir),
    ])
    donor_manifest = donor_dir / "humanml3d_amass_smplx_donor_manifest.json"
    if args.stop_after == "collect":
        return

    plan_output_dir = args.plan_output_dir.resolve()
    run([
        python, "tools/prepare_interx_shape_augmentation_plan.py",
        "--external-donor-manifest", str(donor_manifest),
        "--output-dir", str(plan_output_dir),
        "--seed", str(args.seed),
    ])
    plan_manifest = plan_output_dir / "plan_manifest.json"
    plan_audit = output_root / "plan_audit.json"
    run([
        python, "tools/audit_interx_shape_augmentation_plan.py",
        "--plan", str(plan_manifest), "--output", str(plan_audit),
    ])
    plan = load_json(plan_manifest)
    donors = load_json(donor_manifest)
    completion = {
        "schema": "hhc_humanml3d_amass_shape_pipeline_v1",
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "maximum_transfer_rmse_m": args.maximum_rmse_m,
        "artifacts": {
            "asset_audit": artifact(audit_path),
            "inventory_manifest": artifact(inventory_manifest),
            "transfer_input_manifest": artifact(transfer_manifest_path),
            "donor_manifest": artifact(donor_manifest),
            "augmentation_plan": artifact(plan_manifest),
            "augmentation_plan_audit": artifact(plan_audit),
        },
        "counts": {
            "transferred_source_identities": donors[
                "transferred_source_identity_count"
            ],
            "unique_smplx_donor_shapes": donors["donor_count"],
            "eligible_interx_augmentation_samples": plan["eligible_samples"],
            "positive_contact_timing_samples": plan["positive_timing_samples"],
        },
        "invariants": {
            "humanml3d_train_only": True,
            "official_smplh_to_smplx_transfer": True,
            "direct_beta_copy": False,
            "manual_contact_timing_unchanged": True,
            "interx_motion_caption_and_split_unchanged": True,
        },
    }
    completion_path = output_root / "pipeline_manifest.json"
    atomic_json(completion_path, completion)
    print(json.dumps({
        "status": "complete",
        "manifest": str(completion_path),
        **completion["counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
