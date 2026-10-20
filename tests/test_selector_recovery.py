"""Run with python -m unittest tests.test_selector_recovery."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from tools.select_no_shape_checkpoint_on_official_interx_val import validate_run


class RecoveryTest(unittest.TestCase):
    def test_explicit_missing_candidate_and_wrong_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            resolved = dict(
                total_iter=30000, diagnostic_stop_iter=20000, shape=False,
                contact=False, contact_conditioning=False,
                training_mask_contract="historical_pipeline_paper_kinematic_ratio",
                training_loss_contract="paper_masked_only",
                training_validation_contract="paper_isolated_common",
                sampler="sample_intra_cas", sampling_steps=50, temperature=0.5,
                cfg_scale=3.5, eval_iter=10000, save_all_validation_checkpoints=True,
                wandb_mode="online", seed=1, exp_name="test", out_dir=str(root),
            )
            (root / "training_manifest.json").write_text(json.dumps({"resolved_arguments": resolved}))
            (root / "latest-000020000.ckpt").touch()
            (root / "model-000020000-0.2.ckpt").touch()
            recovery = root / "recovery.ckpt"
            saved = dict(global_step=10000, hyper_parameters={"args": SimpleNamespace(**resolved)})
            torch.save(saved, recovery)
            args = SimpleNamespace(run_dir=root, final_step=20000, interval=10000,
                                   expected_training_loss_contract="paper_masked_only",
                                   recovery_checkpoint=[recovery])
            self.assertEqual(sorted(validate_run(args)[1]), [10000, 20000])
            args.recovery_checkpoint = [recovery, recovery]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_run(args)
            args.recovery_checkpoint = [recovery]
            saved["hyper_parameters"]["args"].out_dir = str(root / "other")
            torch.save(saved, recovery)
            with self.assertRaisesRegex(ValueError, "different run"):
                validate_run(args)


if __name__ == "__main__":
    unittest.main()
