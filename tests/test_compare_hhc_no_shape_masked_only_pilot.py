import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import compare_hhc_no_shape_masked_only_pilot as compare


METRICS = compare.METRICS


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def manifest(loss: str, *, cfg: float = 3.5) -> dict:
    return {
        "resolved_arguments": {
            "shape": False,
            "contact": False,
            "contact_conditioning": False,
            "paper_architecture": False,
            "training_mask_contract": "historical_pipeline_paper_kinematic_ratio",
            "training_loss_contract": loss,
            "optimizer_contract": "historical_multistep",
            "conditioning_contract": "historical_mmm_shape_dropout",
            "training_token_length_contract": "historical_38",
            "training_validation_contract": "paper_isolated_common",
            "sampler": "sample_intra_cas",
            "sampling_steps": 50,
            "temperature": 0.5,
            "cfg_scale": cfg,
            "batch_size": 64,
            "seed": 114514,
            "eval_iter": 10000,
        }
    }


def audit(mm: float) -> dict:
    values = {
        "val/FID": 0.4,
        "val/Matching": 3.2,
        "val/R_precision_top1": 0.45,
        "val/R_precision_top2": 0.65,
        "val/R_precision_top3": 0.76,
        "val/Diversity": 9.0,
        "val/Multimodality_diagnostic": mm,
    }
    return {
        "status": "pass",
        "metrics": {
            metric: {
                "events": [
                    {"wandb_step": 100 * (index + 1), "value": value}
                    for index in range(5)
                ]
            }
            for metric, value in values.items()
        },
    }


def selection(loss: str, *, fid: float = 0.3) -> dict:
    joint = {
        "fid": fid,
        "matching": 3.1,
        "top1": 0.45,
        "top2": 0.66,
        "top3": 0.77,
        "diversity": 9.1,
        "diversity_gt": 9.0,
    }
    return {
        "status": "complete",
        "selected_step": 30000,
        "contract": {"training_loss_contract": loss},
        "candidates": [
            {"step": step, "joint_metrics": {"JointsFull": joint}}
            for step in range(10000, 50001, 10000)
        ],
    }


class CompareMaskedOnlyPilotTest(unittest.TestCase):
    def invoke(self, *, pilot_cfg: float = 3.5, pilot_mm: float = 0.25) -> dict:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        paths = {
            "control_manifest": root / "control_manifest.json",
            "control_wandb_audit": root / "control_audit.json",
            "control_selection": root / "control_selection.json",
            "pilot_manifest": root / "pilot_manifest.json",
            "pilot_wandb_audit": root / "pilot_audit.json",
            "pilot_selection": root / "pilot_selection.json",
            "source_equivalence_audit": root / "source_audit.json",
            "output": root / "comparison.json",
        }
        write(paths["control_manifest"], manifest("historical_all_valid"))
        write(paths["control_wandb_audit"], audit(0.20))
        write(paths["control_selection"], selection("historical_all_valid"))
        write(paths["pilot_manifest"], manifest("paper_masked_only", cfg=pilot_cfg))
        write(paths["pilot_wandb_audit"], audit(pilot_mm))
        write(paths["pilot_selection"], selection("paper_masked_only"))
        write(paths["source_equivalence_audit"], {
            "status": "pass", "behavioral_single_factor": True,
        })
        argv = ["compare"]
        for key, value in paths.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        with mock.patch.object(sys, "argv", argv):
            compare.main()
        return json.loads(paths["output"].read_text(encoding="utf-8"))

    def test_promotes_only_loss_change_with_nonlower_mm(self) -> None:
        result = self.invoke()
        self.assertEqual(result["decision"], "continue_same_masked_only_run_to_300k")
        self.assertTrue(all(result["promotion_screen"].values()))
        self.assertFalse(result["formal_test_metrics_used_for_decision"])

    def test_rejects_hidden_cfg_change(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-loss training factors differ"):
            self.invoke(pilot_cfg=4.0)

    def test_stops_when_mmodality_is_lower(self) -> None:
        result = self.invoke(pilot_mm=0.15)
        self.assertEqual(result["decision"], "do_not_continue_masked_only_arm")
        self.assertFalse(result["promotion_screen"]["mm_not_lower"])


if __name__ == "__main__":
    unittest.main()
