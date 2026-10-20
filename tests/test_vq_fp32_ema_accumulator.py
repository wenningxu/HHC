import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from models.quantize_cnn import QuantizeEMAReset
from tools import train_vq_fp32_ema_accumulator as pilot


class AccumulatorPilotTest(unittest.TestCase):
    def test_stationarity_initialization_and_native_selection(self):
        torch.manual_seed(90210)
        x = (torch.randn(32, 96) * 20).bfloat16()
        q = QuantizeEMAReset(32, 96, SimpleNamespace(mu=0.99))
        original_quantize = QuantizeEMAReset.quantize
        rng = torch.get_rng_state().clone()
        pilot.init_codebook(q, x)
        self.assertTrue(torch.equal(q.codebook, x))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(q.code_sum.dtype, torch.float32)
        ids = torch.arange(32).repeat(4)
        for _ in range(1000):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                pilot.update_codebook(q, x.repeat(4, 1), ids)
        self.assertLess(float((q.codebook - x.float()).norm() / x.float().norm()), .003)
        self.assertIs(QuantizeEMAReset.quantize, original_quantize)

    def test_provenance_and_fresh_only_guard(self):
        with patch.object(sys, 'argv', ['pilot', '--precision', 'bf16']):
            args = pilot.parse_args()
        self.assertEqual(args.ema_precision_intervention, 'fp32_accumulator_native_aggregation_v1')
        self.assertEqual(len(args.ema_intervention_sources), 2)
        with patch.object(sys, 'argv', ['pilot', '--precision', 'bf16', '--resume-pth', 'old.pth']):
            with self.assertRaises(ValueError):
                pilot.parse_args()


if __name__ == '__main__':
    unittest.main()
