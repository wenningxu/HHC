import json
import hashlib
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_mgt_shape_condition_dependence import donor_mapping, finalize
from tools.build_review_results_report import validate_mgt_shape_dependence
from tools.summarize_mgt_shape_condition_dependence import condition_effect, load


def file_sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def test_shape_derangement_is_stable_and_changes_exact_shape():
    names = ["a", "b", "c", "d"]
    signatures = {"a": "shape-a", "b": "shape-a", "c": "shape-b", "d": "shape-b"}
    first, first_hash = donor_mapping(names, signatures)
    second, second_hash = donor_mapping(names, signatures)
    assert first == second
    assert first_hash == second_hash
    assert all(first[name] != name for name in names)
    assert all(signatures[first[name]] != signatures[name] for name in names)


def test_shape_derangement_rejects_an_impossible_imbalanced_sample():
    names = ["a", "b", "c", "d"]
    signatures = {"a": "shape-a", "b": "shape-a", "c": "shape-a", "d": "shape-b"}
    try:
        donor_mapping(names, signatures)
    except ValueError as error:
        assert "too shape-imbalanced" in str(error)
    else:
        raise AssertionError("impossible cross-shape derangement was accepted")


def test_shape_metric_finalization_uses_token_weighting():
    store = {}
    for condition in (
        "requested", "permuted_beta_and_gender",
        "zero_beta_requested_gender",
    ):
        store[condition] = {
            "root_from_fully_masked": {
                "loss_sum": 2.0, "correct": 1, "tokens": 2,
            },
            "body_after_gt_root": {
                "loss_sum": 6.0, "correct": 2, "tokens": 3,
            },
            "hand_after_gt_root_body": {
                "loss_sum": 5.0, "correct": 0, "tokens": 5,
            },
        }
    result = finalize(store)
    aggregate = result["aggregate"]["requested"]
    assert aggregate["tokens"] == 10
    assert aggregate["cross_entropy"] == 1.3
    assert aggregate["accuracy_percent"] == 30.0
    assert result["deltas"]["permuted_beta_and_gender"][
        "accuracy_percent_candidate_minus_requested"
    ] == 0.0


def test_pair_loader_and_effect_are_fail_closed():
    payload = {
        "schema": "hhc_mgt_shape_condition_dependence_audit_v1",
        "status": "complete",
        "split_role": "training",
        "metrics": {
            "aggregate": {
                "requested": {"accuracy_percent": 20.0, "cross_entropy": 2.0},
                "permuted_beta_and_gender": {
                    "accuracy_percent": 10.0, "cross_entropy": 3.0,
                },
            }
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audit.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load(path, "training")
        effect = condition_effect(loaded, "permuted_beta_and_gender")
        assert effect["requested_accuracy_advantage_percent_points"] == 10.0
        assert effect["requested_cross_entropy_advantage"] == 1.0
        try:
            load(path, "validation")
        except ValueError as error:
            assert "expected validation" in str(error)
        else:
            raise AssertionError("wrong split role was accepted")


def test_final_report_shape_gate_rejects_threshold_tampering():
    audit_impl = ROOT / "tools/audit_mgt_shape_condition_dependence.py"
    pair_impl = ROOT / "tools/summarize_mgt_shape_condition_dependence.py"
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        checkpoint = directory / "selected.ckpt"
        checkpoint.write_bytes(b"selected-paper-checkpoint")
        checkpoint_sha = file_sha256(checkpoint)
        records = {
            "hhc_mgt_final_selection": {"payload": {
                "selected_checkpoint_sha256": checkpoint_sha,
            }},
        }
        raw_paths = {}
        for split_role in ("training", "validation"):
            payload = {
                "schema": "hhc_mgt_shape_condition_dependence_audit_v1",
                "status": "complete",
                "split_role": split_role,
                "samples": 256,
                "sample_selection": (
                    "uniform_without_replacement_over_active_split"
                ),
                "shape_derangement": {
                    "fixed_points": 0,
                    "same_shape_signature_pairs": 0,
                    "unique_shape_signatures": 12,
                    "mapping_sha256": "a" * 64,
                },
                "checkpoint": {
                    "path": str(checkpoint), "sha256": checkpoint_sha,
                },
                "training_manifest": {
                    "path": str(checkpoint), "sha256": checkpoint_sha,
                },
                "implementation": {
                    "path": str(audit_impl), "sha256": file_sha256(audit_impl),
                },
                "metrics": {"aggregate": {
                    condition: {
                        "tokens": 100,
                        "cross_entropy": 1.0,
                        "accuracy_percent": 2.0,
                    }
                    for condition in (
                        "requested", "permuted_beta_and_gender",
                        "zero_beta_requested_gender",
                    )
                }},
            }
            raw_path = directory / f"{split_role}.json"
            raw_path.write_text(json.dumps(payload), encoding="utf-8")
            raw_paths[split_role] = raw_path
            records[f"hhc_mgt_shape_dependence_{split_role}"] = {
                "payload": payload,
                "artifact": {
                    "path": str(raw_path), "sha256": file_sha256(raw_path),
                },
            }
        pair = {
            "schema": "hhc_mgt_shape_condition_dependence_pair_v1",
            "status": "pass",
            "decision": "train_only_shape_identity_shortcut_not_established",
            "thresholds_preregistered_before_execution": {
                "minimum_train_requested_shape_advantage_percent_points": 5.0,
                "minimum_train_minus_validation_advantage_percent_points": 3.0,
            },
            "checkpoint": {
                "path": str(checkpoint), "sha256": checkpoint_sha,
            },
            "inputs": {
                role: {"path": str(path), "sha256": file_sha256(path)}
                for role, path in raw_paths.items()
            },
            "implementation": {
                "path": str(pair_impl), "sha256": file_sha256(pair_impl),
            },
        }
        records["hhc_mgt_shape_dependence_pair"] = {"payload": pair}
        validate_mgt_shape_dependence(records)
        training_derangement = records[
            "hhc_mgt_shape_dependence_training"
        ]["payload"]["shape_derangement"]
        training_derangement["same_shape_signature_pairs"] = 1
        try:
            validate_mgt_shape_dependence(records)
        except ValueError as error:
            assert "shape-dependence audit" in str(error)
        else:
            raise AssertionError("same-shape donor pair was accepted")
        training_derangement["same_shape_signature_pairs"] = 0
        pair["thresholds_preregistered_before_execution"][
            "minimum_train_requested_shape_advantage_percent_points"
        ] = 4.0
        try:
            validate_mgt_shape_dependence(records)
        except ValueError as error:
            assert "paired train/validation shape-dependence" in str(error)
        else:
            raise AssertionError("tampered shape shortcut threshold was accepted")


if __name__ == "__main__":
    tests = (
        test_shape_derangement_is_stable_and_changes_exact_shape,
        test_shape_derangement_rejects_an_impossible_imbalanced_sample,
        test_shape_metric_finalization_uses_token_weighting,
        test_pair_loader_and_effect_are_fail_closed,
        test_final_report_shape_gate_rejects_threshold_tampering,
    )
    for test in tests:
        test()
        print({"test": test.__name__, "status": "pass"})
