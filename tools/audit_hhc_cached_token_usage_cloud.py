"""CPU-only descriptive code usage, without loading a model or changing training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summary(counts, positions, adjacent_equal, adjacent_total):
    total = float(counts.sum())
    probabilities = counts[counts > 0] / total
    entropy = float(-(probabilities * probabilities.log()).sum())
    values, indices = counts.topk(10)
    return {'weight_total': total, 'unweighted_observations': positions,
            'active_codes': int((counts > 0).sum()), 'vocabulary': len(counts),
            'entropy_nats': entropy, 'perplexity': math.exp(entropy),
            'top1_share': float(values[0] / total), 'top10_share': float(values.sum() / total),
            'top10': [{'code': int(i), 'share': float(v / total)} for i, v in zip(indices, values)],
            'adjacent_equal_share': adjacent_equal / adjacent_total if adjacent_total else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    if str(ROOT) != '/workspace/HHC' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Cloud CPU only: /workspace/HHC and CUDA_VISIBLE_DEVICES empty required')
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    cache_path = args.cache.resolve()
    assert cache_path.is_relative_to(ROOT / 'output') and cache_path.name == 'tokens.pt'
    with cache_path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        cache = torch.load(stream, map_location='cpu', weights_only=False)
        after = os.fstat(stream.fileno())
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    rows = cache['rows']
    output = {}
    for view in ('full_cached_crops', 'historical_first75_augmentation_expectation'):
        totals = torch.zeros((2, 12288), dtype=torch.float64)
        row_equal = totals.clone()
        positions = 0 if view == 'full_cached_crops' else None
        adjacent_equal = [0., 0.]
        adjacent_total = 0.
        for row in rows:
            token = row['tokens']
            assert token.device.type == 'cpu' and token.ndim == 2 and token.shape[1] == 2
            assert len(token) and int(token.min()) >= 0 and int(token.max()) < 12288
            if view == 'full_cached_crops':
                variants = [(token, 1.)]
            elif len(token) > 1:
                # Exact historical augmentation probabilities; no random draw.
                variants = [(token[:75], 2/3), (token[:-1][:75], 1/6), (token[1:][:75], 1/6)]
            else:
                variants = [(token, 1.)]
            for item, weight in variants:
                if positions is not None:
                    positions += len(item)
                for channel in range(2):
                    counts = torch.bincount(item[:, channel], minlength=12288).double()
                    totals[channel] += weight * counts
                    row_equal[channel] += weight * counts / len(item)
                    adjacent_equal[channel] += weight * int((item[1:, channel] == item[:-1, channel]).sum())
                adjacent_total += weight * max(0, len(item) - 1)
        output[view] = {
            name: {'pooled_positions': summary(totals[channel], positions, adjacent_equal[channel], adjacent_total),
                   'equal_row_marginal': summary(row_equal[channel], len(rows), 0, 0)}
            for channel, name in enumerate(('body', 'root'))}
    assert not torch.cuda.is_initialized()
    print(json.dumps({'schema': 'hhc_cached_code_usage_v1', 'cache': str(cache_path),
                      'cache_bytes': before.st_size, 'cache_sha256': digest,
                      'cache_signature': cache['signature'], 'rows': len(rows),
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'cpu_only_cuda_uninitialized': True, 'views': output,
                      'limits': ['Descriptive training cache only; no val/test or generated samples.',
                                 'Row-equal marginal and pooled-position majority shares are not exact microbatch-averaged accuracy baselines.',
                                 'Code usage does not prove collapse, semantic quality, FID improvement, or SOTA.',
                                 'Root/body have different kinematic masks; all-valid accuracy is not masked-channel accuracy.']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
