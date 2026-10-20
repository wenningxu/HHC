#!/usr/bin/env python
"""Isolate EMA arithmetic with fixed assignments and stationary code centroids."""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.quantize_cnn import QuantizeEMAReset


@torch.inference_mode()
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(90210)
    # Exactly the same representable centroids and assignments in every arm.
    centers = (torch.randn(32, 96, device=device) * 20).bfloat16()
    x = centers.repeat(4, 1)
    ids = torch.arange(32, device=device).repeat(4)
    result = {'diagnostic_only': True, 'synthetic_stationary_input': True,
              'seed': 90210, 'device': str(device), 'updates': 1000,
              'codes': 32, 'dimension': 96, 'observations_per_code': 4,
              'mu': 0.99, 'results': {}}
    for arm in ('native_bf16', 'fp32_accumulator_bf16_matmul', 'full_fp32_update'):
        q = QuantizeEMAReset(32, 96, SimpleNamespace(mu=0.99)).to(device)
        q.init_codebook(centers.clone())
        if arm != 'native_bf16':
            q.code_sum = q.code_sum.float()
        rows = {}
        for step in range(1, 1001):
            enabled = arm != 'full_fp32_update'
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled):
                q.update_codebook(x if enabled else x.float(), ids)
            if step in (1, 100, 1000):
                error = q.codebook.float() - centers.float()
                assert error.isfinite().all()
                rows[step] = {'codebook_rmse': float(error.square().mean().sqrt()),
                              'relative_l2_error': float(error.norm() / centers.float().norm()),
                              'code_sum_dtype': str(q.code_sum.dtype)}
        if arm == 'full_fp32_update':
            assert rows[1000]['relative_l2_error'] < 1e-4
        result['results'][arm] = rows
    print('RESULT=' + json.dumps(result))


if __name__ == '__main__':
    main()
