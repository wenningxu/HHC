#!/usr/bin/env python
"""Fail fast unless the frozen ordinary-FID MGT repair gate has passed."""

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_EVIDENCE_ROLES = {
    "vq_reconstruction",
    "historical_training_trajectory",
    "historical_ordinary_fid",
    "historical_checkpoint_pair",
    "common50_ordinary_fid",
    "historical_launch_sources",
}
VQ_SHA256 = "fdf5afaa5d3513f933254fafc831f6046f7c38b5e649db08fb6be7e0ce1006bb"
CONTACT_MAP_SHA256 = "56109a68bcf5d568cfbfc86de1ba373a8c651a6a4fdbf5007325f60a23809527"
EVALUATOR_OPTIONS_SHA256 = "f4ddaa81157259ce6b795f456be98263e50cd788a7800c30aacbfc2361b2d6dd"
EVALUATOR_CHECKPOINT_SHA256 = "b1699f13276c044d8b7ae5f6063eb241b7ef96d7e92c878857abd5081e40701e"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("output/mgt_repair_gate.json"),
    )
    args = parser.parse_args()

    require(args.gate.is_file(), f"MGT repair gate is missing: {args.gate}")
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    require(gate.get("schema") == "hhc_mgt_repair_gate_v1", "wrong gate schema")
    require(gate.get("status") == "pass", "MGT repair gate has not passed")

    tokenizer = gate.get("tokenizer", {})
    require(tokenizer.get("contract") == "precleanup_raw12288", "wrong VQ contract")
    require(tokenizer.get("checkpoint_iteration") == 175000, "wrong VQ iteration")
    require(tokenizer.get("codebook_size") == 12288, "wrong VQ codebook size")
    require(tokenizer.get("checkpoint_sha256") == VQ_SHA256, "wrong VQ checkpoint hash")
    reconstruction_evaluator = tokenizer.get(
        "reconstruction_evaluator_provenance", {}
    )
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
        record = reconstruction_evaluator.get(name, {})
        path = Path(record.get("path", ""))
        require(path.is_file(), f"VQ reconstruction evaluator artifact is missing: {name}")
        require(path.stat().st_size == record.get("bytes"),
                f"VQ reconstruction evaluator artifact size changed: {name}")
        require(sha256_file(path) == record.get("sha256"),
                f"VQ reconstruction evaluator artifact hash changed: {name}")
    require(Path(reconstruction_evaluator["scorer_source"]["path"]).name
            == "score_fair_interx_main.py",
            "wrong VQ reconstruction main-only scorer")
    reconstruction_runtime = tokenizer.get(
        "reconstruction_evaluator_runtime", {}
    )
    require(
        reconstruction_runtime.get("contract")
        == "pytorch_official_evaluator_single_device_v1"
        and reconstruction_runtime.get("requested_device") == "cpu"
        and reconstruction_runtime.get("device_type") == "cpu"
        and reconstruction_runtime.get("device_index") is None
        and reconstruction_runtime.get("device_name") == "cpu"
        and isinstance(reconstruction_runtime.get("torch_version"), str)
        and reconstruction_runtime.get("torch_version"),
        "wrong VQ reconstruction evaluator runtime",
    )

    replay = gate.get("historical_replay", {})
    require(replay.get("status") == "pass", "historical replay did not pass")
    require(replay.get("completed_step") == 300000, "historical replay is incomplete")
    require(
        replay.get("trajectory_exact_through_step", -1) >= 299999,
        "historical training trajectory is not exact through the final step",
    )
    require(replay.get("launch_source_recovery_status") == "pass",
            "historical launch-source recovery did not pass")
    require(replay.get("launch_source_snapshot_count") == 14,
            "historical launch-source snapshot set is incomplete")
    require(replay.get("reversed_patch_record_count") == 7,
            "historical launch-source patch record is incomplete")
    snapshots = replay.get("launch_source_snapshots", {})
    require(isinstance(snapshots, dict) and len(snapshots) == 14,
            "historical launch-source snapshots are missing")
    for name, record in snapshots.items():
        path = Path(record.get("path", ""))
        require(path.is_file(), f"historical launch source is missing: {name}")
        require(path.stat().st_size == record.get("bytes"),
                f"historical launch source size changed: {name}")
        require(sha256_file(path) == record.get("sha256"),
                f"historical launch source hash changed: {name}")
    for label in ("launch_training_manifest", "selected_session_records"):
        record = replay.get(label, {})
        path = Path(record.get("path", ""))
        require(path.is_file(), f"historical replay artifact is missing: {label}")
        require(path.stat().st_size == record.get("bytes"),
                f"historical replay artifact size changed: {label}")
        require(sha256_file(path) == record.get("sha256"),
                f"historical replay artifact hash changed: {label}")

    bridge = gate.get("common50_bridge", {})
    require(bridge.get("status") == "pass", "common-50 bridge did not pass")
    require(bridge.get("sampler") == "sample_intra_cas", "wrong common-50 sampler")
    require(bridge.get("sampling_steps") == 50, "common-50 bridge is not 50 steps")
    require(bridge.get("temperature") == 0.5, "wrong common-50 temperature")
    require(bridge.get("cfg_scale") == 3.5, "wrong common-50 CFG scale")
    require(bridge.get("contact_guidance") is True, "common-50 contact guidance is disabled")
    require(bridge.get("contact_guidance_scale") == 1.0, "wrong contact-guidance scale")
    require(bridge.get("contact_timing_source") == "request_oracle", "wrong contact timing source")
    require(
        bridge.get("contact_timing_file_sha256") == bridge.get("requests_sha256"),
        "contact timing is not bound to frozen requests",
    )
    require(bridge.get("contact_map_contract") == "paper_formula_train_only_by_frozen_hash", "wrong contact-map contract")
    require(bridge.get("contact_map_sha256") == CONTACT_MAP_SHA256, "wrong contact map")
    require(bridge.get("vq_contract") == "precleanup_raw12288", "wrong bridge VQ contract")
    require(bridge.get("vq_checkpoint_sha256") == VQ_SHA256, "wrong bridge VQ checkpoint")
    require(bridge.get("evaluator") == "official_interx_full_space", "wrong evaluator")
    evaluator = bridge.get("evaluator_provenance", {})
    require(evaluator.get("contract") == "official_interx_full_672d", "wrong evaluator contract")
    require(evaluator.get("options", {}).get("sha256") == EVALUATOR_OPTIONS_SHA256, "wrong evaluator options")
    require(evaluator.get("checkpoint", {}).get("sha256") == EVALUATOR_CHECKPOINT_SHA256, "wrong evaluator checkpoint")
    evaluator_runtime = bridge.get("evaluator_runtime", {})
    require(
        evaluator_runtime.get("contract")
        == "pytorch_official_evaluator_single_device_v1"
        and evaluator_runtime.get("requested_device") == "cpu"
        and evaluator_runtime.get("device_type") == "cpu"
        and evaluator_runtime.get("device_index") is None
        and evaluator_runtime.get("device_name") == "cpu"
        and isinstance(evaluator_runtime.get("torch_version"), str)
        and evaluator_runtime.get("torch_version"),
        "common-50 evaluator runtime is missing or not CPU",
    )
    require(
        evaluator_runtime
        == gate.get("tokenizer", {}).get("reconstruction_evaluator_runtime"),
        "tokenizer reconstruction and common-50 evaluator runtimes differ",
    )
    require(bridge.get("evaluated_samples") == 1696, "wrong common-50 sample count")
    require(
        bridge.get("reference_checkpoint_step")
        == bridge.get("candidate_checkpoint_step"),
        "common-50 bridge checkpoints are not from the same training step",
    )
    require(
        bridge.get("requests_sha256")
        == "01d894c2e95183b2aaa510470819d64cd6c5862beb47b114320e91270d0100d7",
        "common-50 bridge did not use frozen replication-0 requests",
    )
    maximum_fid_difference = bridge.get("maximum_absolute_fid_difference")
    absolute_fid_difference = bridge.get("absolute_fid_difference")
    require(
        isinstance(maximum_fid_difference, (int, float))
        and 0 <= maximum_fid_difference <= 0.02,
        "common-50 FID tolerance is missing or too permissive",
    )
    require(
        isinstance(absolute_fid_difference, (int, float))
        and absolute_fid_difference <= maximum_fid_difference,
        "common-50 ordinary FID did not reproduce the same-step reference",
    )

    evidence = gate.get("evidence", [])
    require(isinstance(evidence, list), "gate evidence must be a list")
    roles = {item.get("role") for item in evidence if isinstance(item, dict)}
    missing_roles = sorted(REQUIRED_EVIDENCE_ROLES - roles)
    require(not missing_roles, f"missing evidence roles: {missing_roles}")
    for item in evidence:
        require(item.get("status") == "pass", f"evidence did not pass: {item.get('role')}")
        path = Path(item["path"])
        require(path.is_file(), f"evidence file is missing: {path}")
        actual = sha256_file(path)
        require(actual == item.get("sha256"), f"evidence hash mismatch: {path}")

    print(json.dumps({
        "status": "pass",
        "gate": str(args.gate.resolve()),
        "historical_completed_step": replay["completed_step"],
        "common50_evaluated_samples": bridge["evaluated_samples"],
        "evidence_roles": sorted(roles),
    }, indent=2))


if __name__ == "__main__":
    main()
