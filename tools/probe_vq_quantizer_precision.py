#!/usr/bin/env python
"""Frozen-snippet inference counterfactual; never updates a codebook or trains."""
import argparse
import hashlib
import json
import os
import sys
from types import MethodType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import default_collate
from dataset.dataset_interx import MotionDatasetV2HHI
from models.quantize_cnn import QuantizeEMAReset
from tools.eval_vq_interx_reconstruction import model_from_checkpoint, sha256_file


def fp32_quantize(self, x):
    with torch.autocast(device_type=x.device.type, enabled=False):
        return QuantizeEMAReset.quantize(self, x.float())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    selection = json.load(open(args.selection))['selection']['val']
    ds = MotionDatasetV2HHI('dataset/interx/splits/val.txt',
                          'dataset/interx/motions/val.h5', 64, 'raw')
    motion, beta, gender = default_collate([ds[i] for i in selection['indices']])[:3]
    motion = motion.reshape(len(selection['indices']), 64, 672)
    assert hashlib.sha256(motion.numpy().tobytes()).hexdigest() == selection['motion_sha256']
    device = torch.device(args.device)
    motion, beta, gender = [x.to(device) for x in (motion, beta, gender)]
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = model_from_checkpoint(torch.load(args.checkpoint, map_location='cpu'), device)
    quantizers = [getattr(model, 'quantizer_' + part) for part in ('upper', 'lower', 'root')]
    before = [q.codebook.clone() for q in quantizers]
    results = {}
    for mode in ('fp32', 'bf16', 'bf16_fp32_distance'):
        if mode == 'bf16_fp32_distance':
            for q in quantizers:
                q.quantize = MethodType(fp32_quantize, q)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=mode != 'fp32'):
            prediction = model(motion, beta, gender, type='full')[0].float()
        assert prediction.isfinite().all()
        results[mode] = {'smooth_l1': float(torch.nn.functional.smooth_l1_loss(prediction, motion))}
        raw_pred = ds.normalizer.backward(prediction).reshape(-1, 64, 56, 12)
        raw_gt = ds.normalizer.backward(motion).reshape(-1, 64, 56, 12)
        results[mode]['raw_rmse'] = float((raw_pred - raw_gt).square().mean().sqrt())
        results[mode]['root_l2_m'] = float((raw_pred[:, :, 55, [0, 1, 2, 6, 7, 8]] -
                                          raw_gt[:, :, 55, [0, 1, 2, 6, 7, 8]]).reshape(-1, 3).norm(dim=-1).mean())
    assert all(torch.equal(q.codebook, old) for q, old in zip(quantizers, before))
    print('RESULT=' + json.dumps({'diagnostic_only': True, 'checkpoint': args.checkpoint,
          'checkpoint_sha256': sha256_file(args.checkpoint), 'device': str(device),
          'input_sha256': selection['motion_sha256'], 'results': results,
          'codebooks_unchanged': True, 'tf32': False}))


if __name__ == '__main__':
    main()
