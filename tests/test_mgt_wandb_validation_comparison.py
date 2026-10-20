import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.recover_historical_mgt_current_validation import parse_blocks


def test_validation_comparison_reports_but_does_not_apply_checkpoint_offset():
    path = ROOT / "tools/compare_mgt_wandb_validation.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))

    assert '"--checkpoint-step-offset"' in source
    assert 'reference = reference_rows.get(step)' in source
    assert 'candidate = candidate_rows.get(step)' in source
    assert '"conceptual_checkpoint_step": step + args.checkpoint_step_offset' in source
    assert '"offset_affects_metric_lookup": False' in source


def test_original38_launcher_persists_all_10k_to_50k_validation_comparisons():
    source = (
        ROOT / "tools/run_hhc_original38_token_window_control.sh"
    ).read_text(encoding="utf-8")
    assert "tools/compare_mgt_wandb_validation.py" in source
    assert "--steps 9999 19999 29999 39999 49999" in source
    assert "--checkpoint-step-offset 1" in source
    assert "hhc_original38_vs_formal50_validation_10k_50k.json" in source


def test_semantics_audit_rejects_pointwise_comparison_of_legacy_best_metrics():
    path = ROOT / "tools/audit_mgt_validation_log_semantics.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))

    assert '"pointwise_comparable_metrics": ["val/FID"]' in source
    assert '"val/Diversity": "best-so-far closeness to real diversity"' in source
    assert '"val/TOP1": "best-so-far maximum"' in source
    assert '"val/Matching": "best-so-far minimum"' in source
    assert "Final comparisons require" in source


def test_historical_stdout_parser_recovers_current_not_best_metrics():
    text = """
    --> Eva. Iter 40000 :,
      FID. 0.3780 ,
      Diversity Real. 9.0364,
      Diversity. 8.8561,
      R_precision_real. [0.40860849 0.60141509 0.71639151],
      R_precision. [0.39504717 0.56544811 0.69221698],
      matching_score_real. 3.5460853576660156,
      matching_score_pred. 3.7800545692443848,
      multimodality. 0.0000
    """
    row = parse_blocks(text)[40000]
    assert row["FID"] == 0.378
    assert row["Diversity"] == 8.8561
    assert row["R_precision_top2"] == 0.56544811
    assert row["Matching"] == 3.7800545692443848


def test_current_metric_recovery_prefers_frozen_replay_log_without_session():
    stdout = """
    --> Eva. Iter 90000 :,
      FID. 0.2705 ,
      Diversity Real. 9.1047,
      Diversity. 9.1700,
      R_precision_real. [0.42158019 0.61084906 0.72405660],
      R_precision. [0.39445755 0.60023585 0.71462264],
      matching_score_real. 3.533825397491455,
      matching_score_pred. 3.596099376678467,
      multimodality. 0.0000
    """
    candidate = {
        "status": "comparison_complete",
        "comparisons": [{
            "trainer/global_step": 89999,
            "reference": {"val/FID": 0.2705},
            "candidate": {
                "val/FID": 0.21,
                "val/Diversity": 9.2,
                "val/TOP1": 0.37,
                "val/TOP2": 0.56,
                "val/TOP3": 0.67,
                "val/Matching": 3.8,
            },
        }],
    }
    replay = {
        "status": "pass",
        "maximum_absolute_difference": 0.0,
        "comparisons": [{"trainer/global_step": 89999}],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        historical = root / "historical.log"
        replay_log = root / "replay.log"
        replay_audit = root / "replay.json"
        comparison = root / "comparison.json"
        session = root / "session.jsonl"
        output = root / "output.json"
        historical.write_text(stdout, encoding="utf-8")
        replay_log.write_text(stdout, encoding="utf-8")
        replay_audit.write_text(json.dumps(replay), encoding="utf-8")
        comparison.write_text(json.dumps(candidate), encoding="utf-8")
        direct_observation = {
            "timestamp": "2026-09-02T00:00:00Z",
            "payload": {"item": {
                "command": ["rg", "gpu1_mgt_known_good_historical_contract_v7_20260829.log"],
                "stdout": stdout,
            }},
        }
        recursive_recovery_output = {
            "timestamp": "2026-09-02T00:01:00Z",
            "payload": {"item": {
                "command": [
                    "python", "recover_historical_mgt_current_validation.py",
                    "--replay-output-log",
                    "gpu1_mgt_known_good_historical_contract_v7_20260829.log",
                ],
                "stdout": stdout,
            }},
        }
        session.write_text(
            json.dumps(direct_observation) + "\n"
            + json.dumps(recursive_recovery_output) + "\n",
            encoding="utf-8",
        )
        subprocess.run([
            sys.executable,
            str(ROOT / "tools/recover_historical_mgt_current_validation.py"),
            "--historical-output-log", str(historical),
            "--replay-validation-audit", str(replay_audit),
            "--replay-output-log", str(replay_log),
            "--codex-session", str(session),
            "--candidate-comparison", str(comparison),
            "--checkpoint-step", "90000",
            "--output", str(output),
        ], check=True, capture_output=True, text=True)
        payload = json.loads(output.read_text(encoding="utf-8"))
        missing_pointwise = subprocess.run([
            sys.executable,
            str(ROOT / "tools/recover_historical_mgt_current_validation.py"),
            "--historical-output-log", str(historical),
            "--replay-validation-audit", str(replay_audit),
            "--candidate-comparison", str(comparison),
            "--checkpoint-step", "90000",
            "--output", str(root / "must-not-exist.json"),
        ], check=False, capture_output=True, text=True)
    assert payload["status"] == "pass"
    assert payload["replay_current_source"] == "frozen_v7_replay_output_log"
    assert payload["artifacts"]["codex_session_live_source"][
        "captured_target_rows"
    ] == 1
    assert len(payload["replay_stdout_evidence"]) == 1
    assert "recover_historical_mgt_current_validation.py" not in (
        payload["replay_stdout_evidence"][0]["command"]
    )
    assert payload["historical_current_metrics"]["R_precision_top3"] == 0.71462264
    assert missing_pointwise.returncode != 0
    assert "best-so-far W&B fields cannot recover current R-Precision" in (
        missing_pointwise.stderr
    )


if __name__ == "__main__":
    test_validation_comparison_reports_but_does_not_apply_checkpoint_offset()
    test_original38_launcher_persists_all_10k_to_50k_validation_comparisons()
    test_semantics_audit_rejects_pointwise_comparison_of_legacy_best_metrics()
    test_historical_stdout_parser_recovers_current_not_best_metrics()
    test_current_metric_recovery_prefers_frozen_replay_log_without_session()
