#!/usr/bin/env python
"""Archived HHC/GT optimizer diagnostic; not a reportable experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py

from tools.export_fair_h5_to_optimizer_npz import RESUME_CONTRACT


CONDITIONS = ("hhc", "gt")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path).resolve()
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(command, dry_run):
    print(json.dumps({"cwd": str(ROOT), "command": command}), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def hdf5_scalar(value):
    """Normalize scalar HDF5 attributes for strict JSON-style comparison."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def validate_hhc_source_contract(requests, generations, input_dir, samples=1696):
    """Prove that formal optimization starts from the no-timing HHC row."""
    requests_sha256 = sha256_file(requests)
    with h5py.File(generations, "r") as source:
        attributes = {}
        for key, value in source.attrs.items():
            value = hdf5_scalar(value)
            if isinstance(value, (str, int, float, bool)):
                attributes[key] = value
    expected_attributes = {
        "contact_guidance": False,
        "contact_timing_source": "none",
        "contact_timing_file": "",
        "contact_timing_file_sha256": "",
        "contact_map": "",
        "contact_map_sha256": "",
        "sampler": "sample_intra_cas",
        "sampling_steps": 50,
        "requests_sha256": requests_sha256,
    }
    mismatches = {
        key: {"actual": attributes.get(key), "expected": expected}
        for key, expected in expected_attributes.items()
        if attributes.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "formal HHC optimization source is not the no-timing common50 "
            f"condition: {json.dumps(mismatches, sort_keys=True)}"
        )

    conversion_path = input_dir / "_conversion_manifest.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    records_path = input_dir / "generation_records.jsonl"
    records_value = conversion.get("generation_records")
    conversion_checks = {
        "schema": conversion.get("schema") == "hhc_fair_h5_to_optimizer_npz_v1",
        "status": conversion.get("status") == "complete",
        "samples": conversion.get("samples") == samples,
        "repeat_id": conversion.get("repeat_id") == 0,
        "requests_sha256": conversion.get("requests_sha256") == requests_sha256,
        "generations_sha256": (
            conversion.get("generations_sha256") == sha256_file(generations)
        ),
        "generation_records": (
            isinstance(records_value, str)
            and Path(records_value).resolve()
            == records_path.resolve()
        ),
        "generation_records_sha256": (
            records_path.is_file()
            and conversion.get("generation_records_sha256")
            == sha256_file(records_path)
        ),
        "generation_hdf5_attributes": (
            conversion.get("generation_hdf5_attributes") == attributes
        ),
        "resume_contract": (
            conversion.get("resume_contract") == RESUME_CONTRACT
        ),
        "converter_implementation": (
            conversion.get("implementation", {}).get("converter")
            == artifact(ROOT / "tools/export_fair_h5_to_optimizer_npz.py")
        ),
        "rotation_implementation": (
            conversion.get("implementation", {}).get(
                "rotation_conversions"
            ) == artifact(ROOT / "utils/rotation_conversions.py")
        ),
    }
    failed = sorted(key for key, passed in conversion_checks.items() if not passed)
    if failed:
        raise ValueError(
            "formal HHC optimizer conversion has stale or mismatched "
            f"provenance: {failed}"
        )
    return attributes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", type=Path,
        default=Path("output/fair_interx_requests/replication_00.npz"),
    )
    parser.add_argument(
        "--hhc-generations", type=Path,
        default=Path(
            "output/fair_eval/"
            "hhc_historical_plus_paper_mask_ratio_no_contact_common50/"
            "replication_00_main.h5"
        ),
    )
    parser.add_argument(
        "--hhc-input-dir", type=Path,
        default=Path(
            "output/optimization_inputs/"
            "hhc_historical_plus_paper_mask_ratio_no_contact_common50_rep0"
        ),
    )
    parser.add_argument(
        "--gt-generations", type=Path,
        default=Path("output/fair_interx_ground_truth/replication_00.h5"),
    )
    parser.add_argument(
        "--contact-dir", type=Path,
        default=Path("dataset/interx/contact_pairs_codex_luna_strict_v3_test"),
    )
    parser.add_argument(
        "--expected-split", type=Path,
        default=Path("dataset/interx/splits/test.txt"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(
            "output/optimization_historical_paper_mask_original_mmm_rep0"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--evaluator-device", default="cpu",
        help=(
            "Official semantic evaluator runtime; CPU matches the frozen "
            "request GT/text embeddings recovered from the launch command."
        ),
    )
    parser.add_argument("--query-chunk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--acknowledge-invalid-hhc-gt-diagnostic",
        action="store_true",
        help=(
            "Explicitly acknowledge that this archived HHC/GT optimization "
            "does not measure the train-only shape-augmentation mechanism and "
            "its FID/foot metrics are not reportable."
        ),
    )
    args = parser.parse_args()
    if not args.dry_run and not args.acknowledge_invalid_hhc_gt_diagnostic:
        raise RuntimeError(
            "HHC/GT formal optimization is archived and invalid for reporting. "
            "Use the train-only random-shape mismatch pipeline; pass the "
            "explicit acknowledgement flag only for a historical diagnostic."
        )

    requests = args.requests.resolve()
    hhc_generations = args.hhc_generations.resolve()
    gt_generations = args.gt_generations.resolve()
    hhc_input = args.hhc_input_dir.resolve()
    contact_dir = args.contact_dir.resolve()
    expected_split = args.expected_split.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "formal_manifest.json"

    required = (
        requests, hhc_generations, gt_generations,
        hhc_input / "generation_records.jsonl",
        hhc_input / "_conversion_manifest.json",
        contact_dir / "_extraction_manifest.json", expected_split,
    )
    if not args.dry_run:
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        hhc_source_attributes = validate_hhc_source_contract(
            requests, hhc_generations, hhc_input,
        )
    else:
        hhc_source_attributes = None

    tools = (
        "tools/export_fair_h5_to_optimizer_npz.py",
        "tools/run_paper_contact_optimization.py",
        "tools/export_optimizer_npz_to_fair_h5.py",
        "tools/score_fair_interx_main.py",
        "tools/score_fair_interx_physical.py",
        "tools/score_volumetric_smpl_npz.py",
        "tools/compare_paired_physical_metrics.py",
    )
    manifest = {
        "schema": "hhc_original_mmm_optimization_formal_v2",
        "status": "running",
        "design": (
            "same_1696_frozen_requests; unmodified original-MMM optimizer; "
            "HHC-M(no timing/no guidance)+optimization and GT+optimization; "
            "official semantic and annotation/mesh physical metrics"
        ),
        "samples": 1696,
        "repeat_id": 0,
        "optimizer": {
            "implementation_contract": "original_mmm_unmodified_numeric_path",
            "iterations": 200, "learning_rate": 0.01,
            "fine_stage_start": 100,
            "signed_distance_backend": "nearest_vertex",
            "backend_scope": (
                "original MMM CUDA Chamfer nearest-vertex signed distance; "
                "VolumetricSMPL is used only by the read-only physical scorer"
            ),
            "query_chunk": args.query_chunk, "seed": args.seed,
        },
        "semantic_evaluator_device": args.evaluator_device,
        "hhc_source_condition": {
            "contact_guidance": False,
            "contact_timing_source": "none",
            "sampler": "sample_intra_cas",
            "sampling_steps": 50,
            "temporal_mask": (
                "derived_from_initial_geometry_not_test_annotation"
            ),
            "verified_hdf5_attributes": hhc_source_attributes,
        },
        "artifacts": ({
            "requests": artifact(requests),
            "hhc_generations": artifact(hhc_generations),
            "gt_generations": artifact(gt_generations),
            "expected_split": artifact(expected_split),
            "contact_manifest": artifact(contact_dir / "_extraction_manifest.json"),
            **{name: artifact(ROOT / name) for name in tools},
        } if not args.dry_run else {}),
        "conditions": {},
    }
    if not args.dry_run:
        atomic_json(manifest_path, manifest)

    gt_input = output_root / "gt_inputs"
    convert_gt = [
        sys.executable, "tools/export_fair_h5_to_optimizer_npz.py",
        "--requests", str(requests), "--generations", str(gt_generations),
        "--output-dir", str(gt_input), "--repeat-id", "0",
    ]
    if args.overwrite:
        convert_gt.append("--overwrite")
    else:
        convert_gt.append("--resume")
    # Always execute the deterministic converter.  Its strict resume path
    # validates every existing NPZ array against the current request/HDF5,
    # fills only missing files, rejects stale files and refreshes the complete
    # content-addressed ledger.  A manifest-only shortcut is insufficient.
    run(convert_gt, args.dry_run)

    condition_inputs = {
        "hhc": (hhc_generations, hhc_input),
        "gt": (gt_generations, gt_input),
    }
    for condition in CONDITIONS:
        source_h5, input_dir = condition_inputs[condition]
        condition_root = output_root / condition
        optimized = condition_root / "optimized_npz"
        optimized_h5 = condition_root / "optimized_main.h5"
        main_metrics = condition_root / "official_main_metrics.json"
        annotation_physical = condition_root / "annotation_physical.json"
        before_mesh = condition_root / "mesh_before"
        after_mesh = condition_root / "mesh_after"
        paired_mesh = condition_root / "paired_mesh_physical.json"
        generation_records = input_dir / "generation_records.jsonl"

        optimize = [
            sys.executable, "tools/run_paper_contact_optimization.py",
            "--motion-dir", str(input_dir),
            "--generation-manifest", str(generation_records),
            "--contact-dir", str(contact_dir),
            "--expected-split", str(expected_split),
            "--output-dir", str(optimized), "--device", args.device,
            "--repeat-id", "0", "--seed", str(args.seed),
            "--query-chunk", str(args.query_chunk),
            "--num-epochs", "200", "--learning-rate", "0.01",
            "--fine-stage-start", "100",
            "--signed-distance-backend", "nearest_vertex",
        ]
        if args.overwrite:
            optimize.append("--overwrite")
        run(optimize, args.dry_run)

        inverse = [
            sys.executable, "tools/export_optimizer_npz_to_fair_h5.py",
            "--requests", str(requests),
            "--source-generations", str(source_h5),
            "--optimizer-dir", str(optimized),
            "--output", str(optimized_h5), "--repeat-id", "0",
            "--resume",
        ]
        if args.overwrite:
            inverse.append("--overwrite")
        run(inverse, args.dry_run)
        run([
            sys.executable, "tools/score_fair_interx_main.py",
            "--requests", str(requests),
            "--main-generations", str(optimized_h5),
            "--output", str(main_metrics), "--device", args.evaluator_device,
            "--batch-size", "32",
        ], args.dry_run)
        run([
            sys.executable, "tools/score_fair_interx_physical.py",
            "--requests", str(requests), "--generations", str(optimized_h5),
            "--output", str(annotation_physical), "--device", args.device,
        ], args.dry_run)
        for folder, output in ((input_dir, before_mesh), (optimized, after_mesh)):
            command = [
                sys.executable, "tools/score_volumetric_smpl_npz.py",
                "--folder", str(folder), "--output-dir", str(output),
                "--device", args.device, "--query-chunk", str(args.query_chunk),
                "--seed", str(args.seed), "--repeat-id", "0",
            ]
            if args.overwrite:
                command.append("--overwrite")
            run(command, args.dry_run)
        run([
            sys.executable, "tools/compare_paired_physical_metrics.py",
            "--before", str(before_mesh), "--after", str(after_mesh),
            "--output", str(paired_mesh), "--seed", str(args.seed),
            "--bootstrap-samples", "10000",
        ], args.dry_run)
        manifest["conditions"][condition] = {
            "source_h5": str(source_h5), "input_npz": str(input_dir),
            "optimized_npz": str(optimized), "optimized_h5": str(optimized_h5),
            "official_main_metrics": str(main_metrics),
            "annotation_physical": str(annotation_physical),
            "mesh_before": str(before_mesh), "mesh_after": str(after_mesh),
            "paired_mesh_physical": str(paired_mesh),
        }
        if not args.dry_run:
            atomic_json(manifest_path, manifest)

    manifest["status"] = "complete"
    if not args.dry_run:
        manifest["output_artifacts"] = {
            condition: {
                name: artifact(path)
                for name, path in {
                    "optimization_completion_manifest": str(
                        Path(record["optimized_npz"])
                        / "_completion_manifest.json"
                    ),
                    "optimized_h5": record["optimized_h5"],
                    "optimized_h5_manifest": record["optimized_h5"] + ".manifest.json",
                    "official_main_metrics": record["official_main_metrics"],
                    "annotation_physical": record["annotation_physical"],
                    "paired_mesh_physical": record["paired_mesh_physical"],
                }.items()
            }
            for condition, record in manifest["conditions"].items()
        }
        atomic_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "status": "complete"}))


if __name__ == "__main__":
    main()
