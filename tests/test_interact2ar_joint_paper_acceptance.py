from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_gate", ROOT / "tools/assess_hhc_against_interact2ar_joint_paper.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metric_directions_and_diversity_distance():
    assert MODULE.compare(0.46, 0.45, "higher")[0]
    assert MODULE.compare(0.20, 0.27, "lower")[0]
    assert MODULE.compare(9.00, 9.30, "closer_to_ground_truth", 8.97)[0]
    assert not MODULE.compare(9.40, 9.30, "closer_to_ground_truth", 8.97)[0]


def test_unconditional_summary_cannot_claim_same_paper_protocol():
    summary = {
        "evaluator_provenance": {
            "generation_conditioning": "unchanged frozen unconditional generations"
        }
    }
    conditioning = MODULE.generation_conditioning(summary)
    assert "unconditional" in conditioning


def test_joint_acceptance_does_not_silently_claim_transition_metrics():
    assert set(MODULE.METRICS) == {
        "R1", "R2", "R3", "FID", "MM_Dist", "Diversity", "MModality"
    }
    assert "PJ" not in MODULE.METRICS
    assert "AUJ" not in MODULE.METRICS
