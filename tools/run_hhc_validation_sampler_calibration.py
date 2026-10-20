#!/usr/bin/env python3
"""Diagnose HHC sampler settings on the frozen 544-record validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GRID = (
    {"temperature": 0.5, "cfg_scale": 3.5},
    {"temperature": 0.5, "cfg_scale": 1.0},
    {"temperature": 0.5, "cfg_scale": 2.5},
    {"temperature": 0.5, "cfg_scale": 4.5},
    {"temperature": 0.3, "cfg_scale": 3.5},
    {"temperature": 0.8, "cfg_scale": 3.5},
)
BRIDGE_MAX_ABSOLUTE_DIFFERENCE = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def label(config: dict) -> str:
    return (
        f"temp{config['temperature']:.1f}_cfg{config['cfg_scale']:.1f}"
    ).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=Path("output/hhc_formal_raw12288_repaired_mgt_selection.json"),
    )
    parser.add_argument(
        "--request-manifest", type=Path,
        default=Path("output/hhc_mgt_validation_bridge_requests/manifest.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("output/hhc_validation_sampler_calibration"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    selection_path = args.selection_manifest.resolve()
    request_manifest_path = args.request_manifest.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    request_manifest = json.loads(
        request_manifest_path.read_text(encoding="utf-8")
    )
    if (
        request_manifest.get("schema") != "hhc_mgt_validation_bridge_request_v1"
        or request_manifest.get("status") != "complete"
        or request_manifest.get("samples") != 544
        or request_manifest.get("seed") != 500123
        or request_manifest.get("test_split_used") is not False
        or request_manifest.get("sampling_order_contract")
        != "exact_hhc_training_validation_loader_batch_and_within_batch_order"
        or request_manifest.get(
            "ordered_sampling_records_match_training_selection"
        ) is not True
        or request_manifest.get(
            "same_record_set_as_common_validation_request"
        ) is not True
    ):
        raise ValueError("frozen validation request contract is incomplete")
    formal_selection = request_manifest.get("formal_hhc_validation_selection", {})
    if len(formal_selection.get("sha256", "")) != 64:
        raise ValueError("validation request does not bind HHC selection records")
    formal_selection_path = Path(formal_selection["path"]).resolve()
    if sha256_file(formal_selection_path) != formal_selection["sha256"]:
        raise ValueError("bound HHC validation-record selection hash changed")

    request_path = Path(request_manifest["request"]["path"]).resolve()
    if sha256_file(request_path) != request_manifest["request"]["sha256"]:
        raise ValueError("frozen validation request hash changed")
    checkpoint = Path(selection["selected_checkpoint"]).resolve()
    if sha256_file(checkpoint) != selection["selected_checkpoint_sha256"]:
        raise ValueError("selected HHC checkpoint hash changed")
    if (
        selection.get("run_complete") is not True
        or selection.get("expected_final_step") != 300000
        or selection.get("test_metrics_used_for_selection") is not False
    ):
        raise ValueError("HHC checkpoint selection is not a completed val-only run")
    expected_validation_contract = {
        "contact_timing_source": "validation_dataset_annotation_oracle",
        "contact_timing_is_test_data": False,
        "contact_as_transformer_training_input": False,
        "sampler": "sample_intra_cas", "sampling_steps": 50,
        "temperature": 0.5, "cfg_scale": 3.5,
        "contact_guidance_scale": 1.0, "contact": True,
        "contact_conditioning": False, "eval_iter": 10000,
    }
    if selection.get("validation_inference_contract") != expected_validation_contract:
        raise ValueError("selected checkpoint validation inference contract changed")

    vq_checkpoint = (ROOT / "checkpoints/vq_precleanup_raw12288/net_175000.pth")
    contact_map = (
        ROOT / "checkpoints/vq_precleanup_raw12288_formal_mgt_v1/"
        "code_contact_probs.npy"
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    payload = {
        "schema": "hhc_validation_sampler_calibration_v1",
        "status": "running",
        "scope": (
            "diagnostic validation-only one-factor calibration; every row uses "
            "50 outer iterations; the request preserves the exact in-training "
            "sampling order; never replaces the user-specified formal row"
        ),
        "sampling_order_contract": request_manifest["sampling_order_contract"],
        "formal_row": {"sampling_steps": 50, "temperature": 0.5,
                       "cfg_scale": 3.5},
        "test_split_used": False,
        "grid": list(GRID),
        "artifacts": {
            "selection_manifest": artifact(selection_path),
            "checkpoint": artifact(checkpoint),
            "request_manifest": artifact(request_manifest_path),
            "requests": artifact(request_path),
            "formal_hhc_validation_selection": artifact(
                formal_selection_path
            ),
            "vq_checkpoint": artifact(vq_checkpoint),
            "contact_map": artifact(contact_map),
            "exporter": artifact(ROOT / "tools/export_hhc_official_interx.py"),
            "scorer": artifact(ROOT / "tools/score_fair_interx_main.py"),
            "implementation": artifact(Path(__file__)),
        },
        "results": {},
    }
    atomic_json(manifest_path, payload)

    seed = int(request_manifest["seed"])
    for config in GRID:
        run_label = label(config)
        # Keep the earlier evaluator-sorted diagnostic artifacts as immutable
        # evidence.  Exact-order bridge runs use distinct filenames.
        generation = output_root / f"exact_order_{run_label}.h5"
        metrics = output_root / f"exact_order_{run_label}.json"
        subprocess.run([
            sys.executable, "tools/export_hhc_official_interx.py",
            "--checkpoint", str(checkpoint),
            "--vq-checkpoint", str(vq_checkpoint),
            "--vq-contract", "precleanup_raw12288",
            "--mgt-contract", "current",
            "--requests", str(request_path),
            "--output", str(generation),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--repeats", "1", "--seed", str(seed), "--resume",
            "--contact-map", str(contact_map),
            "--contact-map-contract", "paper_formula_train_only",
            "--contact-guidance", "--contact-timing-source", "request_oracle",
            "--sampling-steps", "50",
            "--temperature", str(config["temperature"]),
            "--cfg-scale", str(config["cfg_scale"]),
            "--contact-guidance-scale", "1.0",
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable, "tools/score_fair_interx_main.py",
            "--requests", str(request_path),
            "--main-generations", str(generation),
            "--output", str(metrics), "--device", args.device,
            "--batch-size", str(args.batch_size),
        ], cwd=ROOT, check=True)
        score = json.loads(metrics.read_text(encoding="utf-8"))
        payload["results"][run_label] = {
            "configuration": {"sampling_steps": 50, **config},
            "fid": score["fid"], "diversity": score["diversity"],
            "top1": score["top1"], "top2": score["top2"],
            "top3": score["top3"], "matching": score["matching"],
            "generation": artifact(generation), "metrics": artifact(metrics),
        }
        atomic_json(manifest_path, payload)

    formal_label = label({"temperature": 0.5, "cfg_scale": 3.5})
    recorded_fid = float(selection["selected_validation_fid"])
    replayed_fid = float(payload["results"][formal_label]["fid"])
    bridge_difference = abs(replayed_fid - recorded_fid)
    payload["formal_row_bridge"] = {
        "training_validation_fid": recorded_fid,
        "offline_replay_fid": replayed_fid,
        "absolute_difference": bridge_difference,
        "maximum_absolute_difference": BRIDGE_MAX_ABSOLUTE_DIFFERENCE,
        "tolerance_basis": (
            "checkpoint selection records FID to four decimals; standalone "
            "HDF5 re-embedding must reproduce within one 0.0001 unit"
        ),
        "bitwise_equivalence_claimed": False,
        "pass": bridge_difference <= BRIDGE_MAX_ABSOLUTE_DIFFERENCE,
    }
    if bridge_difference > BRIDGE_MAX_ABSOLUTE_DIFFERENCE:
        payload["status"] = "failed_formal_row_bridge"
        atomic_json(manifest_path, payload)
        raise RuntimeError(
            "formal 50-step validation replay does not reproduce checkpoint FID: "
            f"{replayed_fid} vs {recorded_fid}"
        )

    payload["status"] = "complete"
    payload["lowest_validation_fid_row"] = min(
        payload["results"], key=lambda name: payload["results"][name]["fid"]
    )
    atomic_json(manifest_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "lowest_validation_fid_row": payload["lowest_validation_fid_row"],
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
