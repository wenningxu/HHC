#!/usr/bin/env python3
"""Aggregate paper/repository/data/protocol gates for external Inter-X methods."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/external_baseline_reproduction_audit.json"
POLICY_PATH = ROOT / "configs/external_baseline_reproduction_policy.json"
CANONICAL_SPLITS = {
    "train": "47c5c9a09d7293e90e127fb49ba9dc9398afbff66c8f2b45ada3ef6c4453157a",
    "val": "6c77e85017b14b512f42f2d540da2ce05874d9777d707bebd85782d6221f28a8",
    "test": "db32bbe12651875c41113cd20214afcf59ff9489a65aac6f3b0a5b1ecc2b075b",
}
REPOSITORIES = {
    "intermask": {
        "root": Path("/mnt/8T-HDD/wenning/InterMask"),
        "revision": "5100c555de9839b325d0f3d6904669698c5c87f5",
        "origin": "https://github.com/gohar-malik/intermask.git",
        "scope": "official checkpoint; author-native calibration plus common50 replay",
    },
    "coshmdm": {
        "root": Path("/mnt/8T-HDD/wenning/coshmdm-code"),
        "revision": "c5116ab3bead55051ffec2ec9a161a38d2ab5116",
        "origin": "https://github.com/AliManjotho/coshmdm-code.git",
        "scope": (
            "public-repository runnable 262D diffusion Inter-X port under the "
            "user two-day cap; not the paper-described VQ/shape/RL-SDF architecture"
        ),
    },
    "timotion": {
        "root": Path("/mnt/8T-HDD/wenning/TIMotion"),
        "revision": "172580fd7f2551a7cc10fde1f15ff73315742e65",
        "origin": "https://github.com/AIGC-Explorer/TIMotion.git",
        "scope": "public Transformer+LPA Inter-X port; not unreleased RWKV/Mamba",
    },
    "interact2ar": {
        "root": Path("/mnt/8T-HDD/wenning/Interact2Ar"),
        "revision": "a0e053be793ee55414dad458b2d62f3ad9f378f7",
        "origin": "https://github.com/pabloruizponce/Interact2Ar.git",
        "scope": "official checkpoint; GT-first-frame author protocol separate from unconditional fair protocol",
    },
}
SPLIT_PATHS = {
    "intermask": Path("/mnt/8T-HDD/wenning/InterMask/data/Inter-X_Dataset/splits"),
    "coshmdm": Path("/mnt/8T-HDD/wenning/coshmdm-code/data/interx_processed"),
    "timotion": Path("/mnt/8T-HDD/wenning/TIMotion/data/InterX"),
    "interact2ar": Path("/mnt/8T-HDD/wenning/Interact2Ar/data/splits"),
}
EVIDENCE = {
    "intermask": [
        ("output/fair_eval/intermask_official_exact/native20_hdf5_protocol_audit.json",
         "intermask_existing_native20_hdf5_protocol_audit_v1", "pass"),
        ("output/fair_eval/intermask_official_exact_common50/protocol_drift_audit.json",
         "hhc_intermask_common50_protocol_drift_audit_v1", "starts_with_pass"),
        ("output/fair_eval/intermask_official_exact_common50/run_manifest.json", None, None),
        ("output/fair_eval/intermask_official_exact_common50/summary.json", None, None),
    ],
    "coshmdm": [
        ("output/coshmdm_public_release_scope_audit.json",
         "hhc_coshmdm_public_release_scope_audit_v1", "pass"),
        ("output/coshmdm_training_contract_audit_live.json",
         "hhc_coshmdm_interx_training_contract_v1", "pass"),
        ("output/coshmdm_generation_contract_audit.json",
         "hhc_coshmdm_generation_contract_audit_v1", "pass"),
        ("output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/run_manifest.json", None, None),
        ("output/fair_eval/coshmdm_interx_retrained_2daycap_seed123/summary.json", None, None),
    ],
    "timotion": [
        ("output/timotion_interx_training_contract_audit.json",
         "hhc_timotion_interx_training_contract_v1", "pass"),
        ("output/timotion_generation_contract_audit_standard_formal.json",
         "hhc_timotion_generation_contract_audit_v1", "pass"),
        ("output/timotion_interx_formal_wandb_history_audit.json",
         "hhc_wandb_online_metric_history_audit_v1", "pass"),
        ("output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/run_manifest.json", None, None),
        ("output/fair_eval/timotion_transformer_lpa_interx_retrained_fair/summary.json", None, None),
    ],
    "interact2ar": [
        ("output/interact2ar_paper_repo_data_audit.json",
         "interact2ar_paper_repo_data_audit_v1", "starts_with_pass"),
        ("output/fair_eval/interact2ar_official_fair_no_gt_first_frame/run_manifest.json", None, None),
        ("output/fair_eval/interact2ar_official_fair_no_gt_first_frame/summary.json", None, None),
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def artifact_matches(record: dict) -> bool:
    """Validate a file artifact embedded inside another audit."""
    if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
        return False
    path = Path(record["path"])
    return (
        path.is_file()
        and sha256_file(path) == record["sha256"]
        and ("bytes" not in record or path.stat().st_size == record["bytes"])
    )


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    main_table_path = ROOT / "output/current_interx_main_table.json"
    main_table = json.loads(main_table_path.read_text(encoding="utf-8"))
    method_labels = {
        "intermask": "intermask_official_exact_common50",
        "coshmdm": "coshmdm_interx_retrained_2daycap_seed123",
        "timotion": "timotion_transformer_lpa_interx_retrained_fair",
        "interact2ar": "interact2ar_official_fair_no_gt_first_frame",
    }
    methods = {}
    required_gates = {
        "paper_and_supplement_read",
        "official_repository_revision_frozen",
        "paper_vs_release_differences_recorded",
        "official_checkpoint_and_evaluator_assets_verified",
        "dataset_lineage_split_and_filtering_verified",
        "motion_representation_and_normalization_verified",
        "sampling_and_conditioning_verified",
        "author_release_protocol_calibrated_when_available",
        "cross_method_fair_protocol_separately_defined",
        "claim_scope_disclosed",
    }
    checks = {
        "policy_schema": (
            policy.get("schema")
            == "hhc_external_baseline_reproduction_policy_v1"
        ),
        "policy_is_fail_closed": policy.get("fail_closed") is True,
        "policy_covers_every_formal_stage": set(policy.get("applies_before", []))
        == {"training", "generation", "evaluation", "table_admission"},
        "policy_has_complete_required_gates": set(policy.get("required_gates", []))
        == required_gates,
        "policy_separates_author_paper_and_fair_protocols": (
            policy.get("protocol_separation", {}).get("must_not_pool") is True
        ),
        "policy_requires_interact2ar_author_calibration": (
            policy.get("interact2ar", {}).get(
                "author_release_calibration_required"
            ) is True
            and policy.get("interact2ar", {}).get("paper_replications") == 20
            and policy.get("interact2ar", {}).get(
                "released_full_replications"
            ) == 5
            and policy.get("interact2ar", {}).get(
                "released_readme_uses_gt_first_frame"
            ) is True
        ),
    }
    for name, specification in REPOSITORIES.items():
        repo = specification["root"]
        revision = git(repo, "rev-parse", "HEAD")
        origin = git(repo, "remote", "get-url", "origin")
        checks[f"{name}_official_revision"] = revision == specification["revision"]
        checks[f"{name}_official_origin"] = origin == specification["origin"]
        split_records = {}
        for split, expected_hash in CANONICAL_SPLITS.items():
            path = SPLIT_PATHS[name] / f"{split}.txt"
            observed = sha256_file(path)
            checks[f"{name}_{split}_split"] = observed == expected_hash
            split_records[split] = artifact(path)

        evidence_records = []
        evidence_payloads = []
        for relative, schema, status in EVIDENCE[name]:
            path = ROOT / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            if schema is not None:
                checks[f"{name}_{path.stem}_schema"] = payload.get("schema") == schema
            if status == "starts_with_pass":
                checks[f"{name}_{path.stem}_status"] = str(
                    payload.get("status", "")
                ).startswith("pass")
            elif status is not None:
                checks[f"{name}_{path.stem}_status"] = payload.get("status") == status
            evidence_records.append(artifact(path))
            evidence_payloads.append(payload)

        # A passing JSON copied from an earlier run is not sufficient: verify
        # that the source/config/stat artifacts which its contract bound are
        # still the files present in the runnable checkout.
        if name == "coshmdm":
            release_scope, training_contract, generation_contract = evidence_payloads[:3]
            checks["coshmdm_public_release_claim_scope_is_honest"] = (
                release_scope.get("classification")
                == "public_repository_runnable_diffusion_port_not_paper_architecture_reproduction"
                and release_scope.get("released_runnable_pipeline", {}).get(
                    "shape_conditioning"
                ) is False
                and release_scope.get("released_runnable_pipeline", {}).get(
                    "vq_latent_pipeline"
                ) is False
                and release_scope.get("released_runnable_pipeline", {}).get(
                    "rl_sdf_policy"
                ) is False
            )
            checks["coshmdm_public_release_scope_sources_are_current"] = all(
                artifact_matches(record)
                for record in release_scope.get("artifacts", {}).values()
            )
            checks["coshmdm_training_frozen_files_are_current"] = all(
                artifact_matches(record)
                for record in training_contract.get("frozen_files", {}).values()
            )
            checks["coshmdm_training_diff_is_current"] = (
                training_contract.get("repository", {}).get(
                    "tracked_diff_sha256"
                )
                == hashlib.sha256(
                    subprocess.check_output(
                        ["git", "-C", str(repo), "diff", "--binary", "HEAD"]
                    )
                ).hexdigest()
            )
            checks["coshmdm_generation_sources_are_current"] = all(
                artifact_matches(record)
                for record in generation_contract.get("sources", {}).values()
            )
        elif name == "timotion":
            training_contract, generation_contract = evidence_payloads[:2]
            paper_protocol = training_contract.get("paper_protocol", {})
            checks["timotion_latest_paper_protocol_is_explicit"] = (
                paper_protocol.get("source")
                == "https://arxiv.org/html/2408.17135v5"
                and paper_protocol.get("version") == "v5"
                and paper_protocol.get("evaluation_replications") == 20
                and paper_protocol.get("sampling_steps") == 50
                and paper_protocol.get("ddim_eta") == 0.0
                and paper_protocol.get("classifier_free_guidance") == 3.5
                and paper_protocol.get("interx_fid") == {
                    "Transformer": 0.385,
                    "Mamba": 0.348,
                    "RWKV": 0.261,
                }
            )
            checks["timotion_public_release_scope_is_not_overclaimed"] = (
                training_contract.get("reported_method")
                == "public TiMotion Transformer+LPA (not unreleased RWKV/Mamba)"
                and training_contract.get("released_checkpoint_compatibility", {})
                .get("documented_interx_checkpoint_available") is False
                and not training_contract.get("released_checkpoint_compatibility", {})
                .get("tracked_interx_paths")
            )
            file_artifacts = [
                record
                for record in training_contract.get("artifacts", {}).values()
                if Path(record.get("path", "")).is_file()
            ]
            checks["timotion_training_file_artifacts_are_current"] = (
                bool(file_artifacts)
                and all(artifact_matches(record) for record in file_artifacts)
            )
            checks["timotion_training_tree_audit_was_independent"] = (
                training_contract.get("tree_contents_independently_rehashed")
                is True
            )
            checks["timotion_generation_sources_are_current"] = all(
                artifact_matches(record)
                for record in generation_contract.get("sources", {}).values()
            )
        elif name == "interact2ar":
            interact_audit = evidence_payloads[0]
            checks["interact2ar_official_sources_match_frozen_revision"] = (
                interact_audit
                .get("repository", {})
                .get("source_match", {})
                .get("pass")
                is True
            )
            interact_paper = interact_audit.get("paper_contract", {})
            checks["interact2ar_latest_paper_metric_contract_is_explicit"] = (
                interact_paper.get("source")
                == "https://arxiv.org/html/2512.19692v2"
                and interact_paper.get("version_date") == "2026-03-27"
                and interact_paper.get("replications") == 20
                and interact_paper.get("autoregressive_sampling_steps") == 10
                and interact_paper.get("joint_evaluators")
                == ["JointsFull", "JointsBody", "JointsHands"]
                and interact_paper.get("shape_policy")
                == "neutral body configuration"
            )
            checks["interact2ar_local_data_are_byte_identical_across_consumers"] = all(
                len({
                    record.get("sha256")
                    for record in interact_audit.get("motions", {})
                    .get(split, {}).get("artifacts", {}).values()
                }) == 1
                for split in ("train", "val", "test")
            )
            checks["interact2ar_official_splits_match_canonical_hashes"] = all(
                interact_audit.get("splits", {}).get(split, {})
                .get("artifacts", {}).get("interx_official_checkout", {})
                .get("sha256") == expected_sha
                for split, expected_sha in CANONICAL_SPLITS.items()
            )
            checks["interact2ar_release_paper_mismatches_are_disclosed"] = (
                interact_audit.get("released_code_contract", {})
                .get("released_full_replications") == 5
                and interact_audit.get("released_code_contract", {})
                .get("readme_author_command_uses_first_frame") is True
                and {entry.get("field") for entry in
                     interact_audit.get("release_paper_mismatches", [])}
                == {"replications", "training batch size / README device count"}
            )
            checks["interact2ar_official_model_and_three_evaluators_are_hashed"] = (
                set(interact_audit.get("official_checkpoints", {}))
                == {"model", "JointsFull", "JointsBody", "JointsHands"}
                and all(
                    len(record.get("sha256", "")) == 64
                    and Path(record.get("path", record.get("resolved_path", ""))).is_file()
                    for record in interact_audit.get("official_checkpoints", {}).values()
                )
            )
        elif name == "intermask":
            native = evidence_payloads[0]
            protocol_drift = evidence_payloads[1]
            common50_run = evidence_payloads[2]
            checks["intermask_paper_main_vs_mm_replications_disclosed"] = (
                native.get("classification")
                == "author-native-20-steps_exact-length_main20_mm20-extension_not-common50"
                and native.get("paper_replication_policy")
                == {
                    "fid_matching_rprecision_diversity": 20,
                    "multimodality": 5,
                }
                and native.get("generated_replication_policy")
                == {
                    "fid_matching_rprecision_diversity": 20,
                    "multimodality": 20,
                }
            )
            checks["intermask_latest_paper_source_is_explicit"] = (
                native.get("paper_source") == {
                    "url": "https://arxiv.org/html/2410.10010v3",
                    "version": "v3",
                    "revised": "2025-03-02",
                    "reported_interx_fid": 0.399,
                }
            )
            checks["intermask_common50_protocol_drift_is_disclosed"] = (
                str(protocol_drift.get("status", "")).startswith("pass")
                and all(protocol_drift.get("checks", {}).values())
            )
            checks["intermask_common50_execution_snapshots_are_intact"] = all(
                artifact_matches(record)
                for record in common50_run.get("provenance", {})
                .get("implementation", {})
                .values()
            )

        label = method_labels[name]
        summary = main_table["methods"][label]
        checks[f"{name}_main_table_replication_scope"] = (
            main_table["protocol"]["replications"] == 20
            and main_table["protocol"]["samples_per_replication"] == 1696
            and len(summary["metrics"]["fid"]["values"]) == 20
        )
        methods[name] = {
            "repository": {
                "path": str(repo), "revision": revision, "origin": origin,
                "tracked_diff_sha256": hashlib.sha256(
                    subprocess.check_output(
                        ["git", "-C", str(repo), "diff", "--binary", "HEAD"]
                    )
                ).hexdigest(),
            },
            "paper_primary_source": {
                "intermask": "https://arxiv.org/html/2410.10010v3",
                "coshmdm": "https://doi.org/10.1109/TVCG.2026.3675725",
                "timotion": "https://arxiv.org/html/2408.17135v5",
                "interact2ar": "https://arxiv.org/html/2512.19692v2",
            }[name],
            "paper_access_limitation": (
                "DOI and project abstract available; project Paper/arXiv buttons "
                "disabled and IEEE full text was not retrievable in the audited environment"
                if name == "coshmdm" else None
            ),
            "allowed_claim_scope": specification["scope"],
            "official_interx_splits": split_records,
            "main_table": {
                "label": label,
                "display_name": summary["display_name"],
                "claim_scope": summary["claim_scope"],
                "fid": summary["metrics"]["fid"],
            },
            "evidence": evidence_records,
        }

    payload = {
        "schema": "hhc_external_baseline_reproduction_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "policy": (
            "A method may enter the fair table only after its paper/project source, "
            "official repository revision, data split hashes, checkpoint availability, "
            "representation adapter and sampling/evaluator scope are disclosed. A local "
            "port must not be relabeled as a native or paper-exact reproduction."
        ),
        "machine_policy": policy,
        "canonical_interx_split_sha256": CANONICAL_SPLITS,
        "methods": methods,
        "artifacts": {
            "main_table": artifact(main_table_path),
            "human_readable_audit": artifact(
                ROOT / "docs/baseline_availability_audit.md"
            ),
            "implementation": artifact(Path(__file__)),
            "machine_policy": artifact(POLICY_PATH),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"], "checks": len(checks),
        "methods": list(methods), "output": str(OUTPUT),
    }, indent=2))
    if payload["status"] != "pass":
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"external baseline audit failed: {failed}")


if __name__ == "__main__":
    main()
