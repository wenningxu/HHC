import unittest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from models.vqvae_sep3_shape import VQVAE_SEP3_shape
from utils.fk import FK_Pair_Layer_Batch
from utils.paramUtil import INTERX_SKELETONS, SMPLX_TREE


class TemporalStrideTest(unittest.TestCase):
    def test_training_rejects_bad_stride_before_run_creation(self):
        for args, expected in [(['--stride-t', '3'], 'requires stride_t=2'),
                               (['--down-t', '3', '--window-size', '65'], 'positive multiple')]:
            result = subprocess.run(
                [sys.executable, 'train/train_vq_interx.py', '--wandb-mode', 'disabled', *args],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected, result.stderr)

    def test_fk_does_not_mutate_shared_skeleton(self):
        skeleton = torch.from_numpy(INTERX_SKELETONS.copy())
        original = skeleton.clone()
        first = FK_Pair_Layer_Batch(SMPLX_TREE, skeleton[:, 0], skeleton[:, 1], 'cpu')
        second = FK_Pair_Layer_Batch(SMPLX_TREE, skeleton[:, 0], skeleton[:, 1], 'cpu')
        self.assertTrue(torch.equal(original, skeleton))
        self.assertTrue(torch.equal(first.unified_positions_lookup.weight,
                                    second.unified_positions_lookup.weight))

    def test_real_vq_encode_decode_and_backward_stride4_and8(self):
        args = SimpleNamespace(dataname='hhi', motion_rep='raw', device='cpu', mu=0.99)
        for down in (2, 3):
            model = VQVAE_SEP3_shape(args, nb_code=16, code_dim=32, down_t=down,
                                     stride_t=2, width=16, depth=1)
            self.assertEqual(model.temporal_stride, 2 ** down)
            motion = torch.randn(1, 32, 672)
            beta, gender = torch.zeros(1, 20), torch.zeros(1, 2, dtype=torch.long)
            out, commit, _ = model(motion, beta, gender)
            self.assertEqual(out.shape, motion.shape)
            (out.square().mean() + commit).backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
            model.eval()
            with torch.no_grad():
                ids = model(motion, beta, gender, type='encode')
                self.assertEqual(ids.shape, (1, 32 // (2 ** down), 3))
                decoded = model(ids, beta, gender, type='decode')
                self.assertEqual(decoded.shape, motion.shape)
                self.assertTrue(torch.isfinite(decoded).all())


if __name__ == '__main__':
    unittest.main()
