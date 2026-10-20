#!/usr/bin/env python
"""Isolated pilot: FP32 EMA accumulator, native aggregation and token selection."""
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.quantize_cnn import QuantizeEMAReset
import options.option_vq as option_vq

_INIT = QuantizeEMAReset.init_codebook
_UPDATE = QuantizeEMAReset.update_codebook
_PARSE = option_vq.get_args_parser


def init_codebook(self, x):
    _INIT(self, x)
    self.code_sum = self.code_sum.float()


def update_codebook(self, x, code_idx):
    self.code_sum = self.code_sum.float()
    return _UPDATE(self, x, code_idx)


def parse_args():
    args = _PARSE()
    if args.resume_pth:
        raise ValueError('This causal pilot requires a fresh matched run, not resume.')
    if args.precision != 'bf16':
        raise ValueError('The accumulator-only pilot requires bf16 outer precision.')
    args.ema_precision_intervention = 'fp32_accumulator_native_aggregation_v1'
    args.ema_intervention_sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), ROOT / 'models/quantize_cnn.py')
    }
    return args


def main():
    # Process-local overrides keep the live baseline source and all callers intact.
    QuantizeEMAReset.init_codebook = init_codebook
    QuantizeEMAReset.update_codebook = update_codebook
    option_vq.get_args_parser = parse_args
    runpy.run_path(str(ROOT / 'train/train_vq_interx.py'), run_name='__main__')


if __name__ == '__main__':
    main()
