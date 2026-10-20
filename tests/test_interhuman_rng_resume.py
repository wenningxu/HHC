"""Run on CUDA to reproduce checkpoint map_location moving RNG byte states."""
import importlib

import torch


def test_rng_restore_after_checkpoint_device_mapping():
    for name in ('train.train_vq_interhuman', 'train.train_mgt_interhuman'):
        module = importlib.import_module(name)
        original = module.capture_rng_state()
        try:
            expected_cpu = torch.rand(9)
            expected_cuda = torch.rand(9, device='cuda:0') if torch.cuda.is_available() else None
            mapped = dict(original)
            if torch.cuda.is_available():
                mapped['torch'] = mapped['torch'].to('cuda:0')
                mapped['cuda'] = [value.to('cuda:0') for value in mapped['cuda']]
            module.restore_rng_state(mapped)
            assert torch.equal(torch.rand(9), expected_cpu)
            if expected_cuda is not None:
                assert torch.equal(torch.rand(9, device='cuda:0'), expected_cuda)
        finally:
            module.restore_rng_state(original)
