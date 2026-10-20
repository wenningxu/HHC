#!/usr/bin/env python
"""Build the MGT repair gate only from validated, content-bound evidence."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


VQ_SHA256 = "fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb"
CONTACT_MAP_SHA256 = "56109a68bcf5d568cfbfc86de1ba373a8c651a6a4fdbf5007325f60a23809527"
EVALUATOR_OPTIONS_SHA256 = "f4ddaa81157259ce6b795f456be98263e50cd788a7800c30aacbfc2361b2d6dd"
EVALUATOR_CHECKPOINT_SHA256 = "b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e"
REQUIRED_VALIDATION_STEPS = list(range(9999, 300000, 10000))
LAUNCH_SOURCE_SCHEMA = "hhc_mgt_launch_source_recovery_v1"
LAUNCH_SOURCE_COUNT = 14
LAUNCH_PATCH_LINES = {68502, 68508, 68514, 68572, 68683, 68689, 68854}
LAUNCH_CHANGED_SOURCES = {
    "options/option_transformer.py",
    "train/train_t2i_trans.py",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def evidence(role: str, path: Path) -> dict:
    record = load(path)
    require(record.get("status") in ("pass", "complete"), f"{role} did not pass")
    return {
        "role": role,
        "status": "pass",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def validate_artifact(record: dict, label: str) -> Path:
    require(isinstance(record, dict), f"{label} is not an artifact record")
    path = Path(record.get("path", ""))
    require(path.is_file(), f"{label} is missing: {path}")
    require(path.stat().st_size == record.get("bytes"), f"{label} byte size changed")
    require(sha256_file(path) == record.get("sha256"), f"{label} hash changed")
    return path


def validate_launch_source_recovery(path: Path) -> dict:
    recovery = load(path)
    require(recovery.get("schema") == LAUNCH_SOURCE_SCHEMA, "wrong launch-source recovery schema")
    require(recovery.get("status") == "pass", "launch-source recovery did not pass")
    require(recovery.get("all_snapshot_hashes_match_launch_manifest") is True,
            "launch-source snapshots were not proven against the manifest")
    require(set(recovery.get("changed_sources_recovered_from_history", []))
            == LAUNCH_CHANGED_SOURCES, "wrong recovered launch-source set")

    manifest_path = validate_artifact(
        recovery.get("training_manifest", {}), "launch training manifest"
    )
    selected_path = validate_artifact(
        recovery.get("selected_session_records", {}),
        "selected Codex patch records",
    )
    selected_lines = selected_path.read_text(encoding="utf-8").splitlines(keepends=True)
    require(len(selected_lines) == len(LAUNCH_PATCH_LINES),
            "selected Codex patch-record count changed")

    patch_records = recovery.get("reversed_patch_records", [])
    require(isinstance(patch_records, list) and len(patch_records) == 7,
            "wrong launch-source patch-record count")
    require({record.get("line") for record in patch_records} == LAUNCH_PATCH_LINES,
            "wrong launch-source patch line set")
    selected_hashes = {
        hashlib.sha256(line.encode("utf-8")).hexdigest()
        for line in selected_lines
    }
    require(
        {record.get("session_line_sha256") for record in patch_records}
        == selected_hashes,
        "selected Codex lines do not match launch-source patch records",
    )
    require(all(record.get("patch_sha256") for record in patch_records),
            "launch-source patch hash is missing")

    manifest = load(manifest_path)
    expected = {
        name: digest for name, digest in manifest.get("sha256", {}).items()
        if name.endswith(".py") and "/" in name
    }
    require(len(expected) == LAUNCH_SOURCE_COUNT,
            "launch manifest does not contain exactly 14 Python sources")
    snapshots = recovery.get("snapshots", {})
    require(isinstance(snapshots, dict) and set(snapshots) == set(expected),
            "launch-source snapshot set differs from the manifest")
    normalized_snapshots = {}
    for name, expected_hash in sorted(expected.items()):
        snapshot_path = validate_artifact(
            snapshots[name], f"launch-source snapshot {name}"
        )
        require(snapshots[name].get("sha256") == expected_hash,
                f"launch-source snapshot hash differs from manifest: {name}")
        normalized_snapshots[name] = {
            "path": str(snapshot_path.resolve()),
            "bytes": snapshot_path.stat().st_size,
            "sha256": expected_hash,
        }
    return {
        "status": "pass",
        "training_manifest": recovery["training_manifest"],
        "selected_session_records": recovery["selected_session_records"],
        "reversed_patch_record_count": len(patch_records),
        "source_snapshot_count": len(normalized_snapshots),
        "source_snapshots": normalized_snapshots,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vq-reconstruction", type=Path, required=True)
    parser.add_argument("--trajectory-audit", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-pair-audit", type=Path, required=True)
    parser.add_argument("--common50-bridge", type=Path, required=True)
    parser.add_argument("--launch-source-recovery", type=Path, required=True)
    parser.add_argument("--scorer-migration-audit", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/mgt_repair_gate.json"))
    args = parser.parse_args()

    launch_sources = validate_launch_source_recovery(args.launch_source_recovery)

    vq = load(args.vq_reconstruction)
    require(vq.get("schema") == "fair_interx_main_only_metrics_v1", "wrong VQ metric schema")
    require(vq.get("status") == "complete" and vq.get("samples") == 1696, "incomplete VQ reconstruction")
    require(math.isfinite(vq.get("fid", math.nan)) and vq["fid"] <= 0.35, "VQ reconstruction FID regressed")
    require(vq.get("main_generation_metadata", {}).get("checkpoint_sha256") == VQ_SHA256, "wrong VQ checkpoint")
    reconstruction_evaluator = vq.get("evaluator_provenance", {})
    require(reconstruction_evaluator.get("contract") == "official_interx_full_672d",
            "wrong VQ reconstruction evaluator contract")
    require(reconstruction_evaluator.get("options", {}).get("sha256")
            == EVALUATOR_OPTIONS_SHA256,
            "wrong VQ reconstruction evaluator options")
    require(reconstruction_evaluator.get("checkpoint", {}).get("sha256")
            == EVALUATOR_CHECKPOINT_SHA256,
            "wrong VQ reconstruction evaluator checkpoint")
    for name in (
        "options", "checkpoint", "wrapper_source", "option_loader_source",
        "scorer_source",
    ):
        validate_artifact(
            reconstruction_evaluator.get(name, {}),
            f"VQ reconstruction evaluator {name}",
        )
    require(
        Path(reconstruction_evaluator["scorer_source"]["path"]).name
        == "score_fair_interx_main.py",
        "VQ reconstruction was not scored by the official main-only scorer",
    )
    reconstruction_runtime = vq.get("evaluator_runtime", {})
    require(
        reconstruction_runtime.get("contract")
        == "pytorch_official_evaluator_single_device_v1"
        and reconstruction_runtime.get("requested_device") == "cpu"
        and reconstruction_runtime.get("device_type") == "cpu"
        and reconstruction_runtime.get("device_index") is None
        and reconstruction_runtime.get("device_name") == "cpu"
        and isinstance(reconstruction_runtime.get("torch_version"), str)
        and reconstruction_runtime.get("torch_version"),
        "VQ reconstruction evaluator runtime is missing or not the frozen CPU contract",
    )

    trajectory = load(args.trajectory_audit)
    require(trajectory.get("schema") == "hhc_mgt_wandb_training_trajectory_comparison_v1", "wrong trajectory schema")
    require(trajectory.get("status") == "pass", "training trajectory did not pass")
    require(
        trajectory.get("comparison_scope")
        == "all_candidate_training_rows_present_in_reference",
        "final trajectory audit did not compare every logged candidate row",
    )
    require(
        trajectory.get("compared_step_count", 0) >= 6000,
        "final trajectory audit contains fewer than 6000 logged steps",
    )
    require(299999 in trajectory.get("requested_steps", []), "final training step was not compared")
    require(trajectory.get("maximum_absolute_loss_difference") == 0, "trajectory tolerance was not zero")
    require(not trajectory.get("threshold_failures") and not trajectory.get("missing"), "trajectory evidence is incomplete")

    validation = load(args.validation_audit)
    require(validation.get("schema") == "hhc_mgt_wandb_validation_comparison_v1", "wrong validation schema")
    require(validation.get("status") == "pass", "historical ordinary FID did not pass")
    require(validation.get("requested_steps") == REQUIRED_VALIDATION_STEPS, "not all validation milestones were compared")
    require(validation.get("maximum_absolute_difference") == 0, "validation tolerance was not zero")
    require(not validation.get("threshold_failures") and not validation.get("missing"), "validation evidence is incomplete")

    checkpoint_pair = load(args.checkpoint_pair_audit)
    require(checkpoint_pair.get("schema") == "hhc_mgt_historical_checkpoint_pair_v1", "wrong checkpoint-pair schema")
    require(checkpoint_pair.get("status") == "pass", "same-step checkpoint pair did not pass")
    require(checkpoint_pair.get("expected_step") == 280000, "checkpoint pair must use step 280000")
    require(all(checkpoint_pair.get("exact", {}).values()), "checkpoint training state is not bitwise exact")

    bridge = load(args.common50_bridge)
    require(bridge.get("schema") == "hhc_mgt_common50_bridge_v1", "wrong common-50 bridge schema")
    require(bridge.get("status") == "pass", "common-50 ordinary FID did not pass")
    require(bridge.get("evaluated_samples") == 1696, "wrong common-50 sample count")
    require(bridge.get("sampling_steps") == 50 and bridge.get("sampler") == "sample_intra_cas", "wrong common-50 sampler")
    require(bridge.get("temperature") == 0.5 and bridge.get("cfg_scale") == 3.5, "wrong common-50 settings")
    require(
        bridge.get("training_conditioning_contract")
        == "historical_mmm_shape_dropout"
        and bridge.get("inference_conditioning_contract") == "paper_text_cfg",
        "common-50 training/inference conditioning contracts are not separated",
    )
    require(bridge.get("contact_guidance") is True, "common-50 bridge did not use contact guidance")
    require(bridge.get("contact_guidance_scale") == 1.0, "wrong common-50 contact-guidance scale")
    require(bridge.get("contact_timing_source") == "request_oracle", "wrong common-50 contact timing")
    require(bridge.get("contact_timing_file_sha256") == bridge.get("requests_sha256"), "contact timing is not bound to requests")
    require(bridge.get("contact_map_contract") == "paper_formula_train_only_by_frozen_hash", "wrong contact-map contract")
    require(bridge.get("contact_map_sha256") == CONTACT_MAP_SHA256, "wrong contact map")
    require(bridge.get("vq_contract") == "precleanup_raw12288", "wrong bridge VQ contract")
    require(bridge.get("vq_checkpoint_sha256") == VQ_SHA256, "wrong bridge VQ checkpoint")
    evaluator = bridge.get("evaluator_provenance", {})
    require(evaluator.get("contract") == "official_interx_full_672d", "wrong bridge evaluator")
    require(evaluator.get("options", {}).get("sha256") == EVALUATOR_OPTIONS_SHA256, "wrong evaluator options")
    require(evaluator.get("checkpoint", {}).get("sha256") == EVALUATOR_CHECKPOINT_SHA256, "wrong evaluator checkpoint")
    bridge_runtime = bridge.get("evaluator_runtime", {})
    require(
        bridge_runtime.get("contract")
        == "pytorch_official_evaluator_single_device_v1"
        and bridge_runtime.get("requested_device") == "cpu"
        and bridge_runtime.get("device_type") == "cpu"
        and bridge_runtime.get("device_index") is None
        and bridge_runtime.get("device_name") == "cpu"
        and isinstance(bridge_runtime.get("torch_version"), str)
        and bridge_runtime.get("torch_version"),
        "common-50 bridge evaluator runtime is missing or not CPU",
    )
    require(
        bridge_runtime == reconstruction_runtime,
        "VQ reconstruction and common-50 bridge evaluator runtimes differ",
    )
    require(bridge["reference"]["checkpoint_step"] == bridge["candidate"]["checkpoint_step"] == 280000, "bridge is not same-step")
    require(bridge["candidate"]["checkpoint_sha256"] == checkpoint_pair["candidate"]["sha256"], "bridge used another candidate checkpoint")
    fid_difference = bridge["differences"]["fid"]["absolute"]
    tolerance = bridge["maximum_absolute_fid_difference"]
    require(0 <= tolerance <= 0.02 and fid_difference <= tolerance, "common-50 FID mismatch")

    evidence_records = [
        evidence("vq_reconstruction", args.vq_reconstruction),
        evidence("historical_training_trajectory", args.trajectory_audit),
        evidence("historical_ordinary_fid", args.validation_audit),
        evidence("historical_checkpoint_pair", args.checkpoint_pair_audit),
        evidence("common50_ordinary_fid", args.common50_bridge),
        evidence("historical_launch_sources", args.launch_source_recovery),
    ]
    if args.scorer_migration_audit is not None:
        migration = load(args.scorer_migration_audit)
        require(
            migration.get("schema")
            == "hhc_vq_reconstruction_scorer_migration_audit_v1"
            and migration.get("status") == "pass",
            "VQ reconstruction scorer migration audit did not pass",
        )
        require(
            migration.get("checks", {}).get("all_metrics_exact") is True
            and migration.get("checks", {}).get("current_scorer_source_bound")
            is True,
            "VQ scorer migration was not exact and source-bound",
        )
        migrated_after = migration.get("artifacts", {}).get("after", {})
        require(
            Path(migrated_after.get("path", "")).resolve()
            == args.vq_reconstruction.resolve()
            and migrated_after.get("sha256")
            == sha256_file(args.vq_reconstruction),
            "VQ scorer migration does not bind the selected reconstruction result",
        )
        evidence_records.append(
            evidence("vq_reconstruction_scorer_migration", args.scorer_migration_audit)
        )

    payload = {
        "schema": "hhc_mgt_repair_gate_v1",
        "status": "pass",
        "tokenizer": {
            "contract": "precleanup_raw12288",
            "checkpoint_iteration": 175000,
            "codebook_size": 12288,
            "checkpoint_sha256": VQ_SHA256,
            "ordinary_reconstruction_fid": vq["fid"],
            "reconstruction_evaluator_provenance": reconstruction_evaluator,
            "reconstruction_evaluator_runtime": reconstruction_runtime,
        },
        "historical_replay": {
            "status": "pass",
            "completed_step": 300000,
            "trajectory_exact_through_step": 299999,
            "ordinary_validation_milestones": len(REQUIRED_VALIDATION_STEPS),
            "launch_source_recovery_status": launch_sources["status"],
            "launch_source_snapshot_count": launch_sources["source_snapshot_count"],
            "launch_source_snapshots": launch_sources["source_snapshots"],
            "launch_training_manifest": launch_sources["training_manifest"],
            "selected_session_records": launch_sources["selected_session_records"],
            "reversed_patch_record_count": launch_sources["reversed_patch_record_count"],
        },
        "common50_bridge": {
            "status": "pass",
            "sampler": "sample_intra_cas",
            "sampling_steps": 50,
            "temperature": 0.5,
            "cfg_scale": 3.5,
            "contact_guidance": True,
            "contact_guidance_scale": 1.0,
            "contact_timing_source": "request_oracle",
            "contact_timing_file_sha256": bridge["contact_timing_file_sha256"],
            "contact_map_contract": bridge["contact_map_contract"],
            "contact_map_sha256": bridge["contact_map_sha256"],
            "vq_contract": bridge["vq_contract"],
            "vq_checkpoint_sha256": bridge["vq_checkpoint_sha256"],
            "evaluator": "official_interx_full_space",
            "evaluator_provenance": evaluator,
            "evaluator_runtime": bridge_runtime,
            "evaluated_samples": 1696,
            "reference_checkpoint_step": 280000,
            "candidate_checkpoint_step": 280000,
            "requests_sha256": bridge["requests_sha256"],
            "reference_fid": bridge["reference"]["fid"],
            "candidate_fid": bridge["candidate"]["fid"],
            "absolute_fid_difference": fid_difference,
            "maximum_absolute_fid_difference": tolerance,
        },
        "evidence": evidence_records,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
