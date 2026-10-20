"""Replay the real released InterMask480 cache through the new native scorer.

No generation or test scoring. Quantifies raw<->normalized float32 roundtrip
error against the already accepted per-request official CPU embeddings.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from models.interhuman_evaluator_wrapper import load_official_interclip, load_official_metrics_module
from tools.score_interhuman_native_baselines_cpu import score_main
from tools.score_fair_interhuman_main import main_metrics_from_embeddings
from tools.hhc_text_only_formal_contract import file_sha


def main():
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2); torch.set_num_interop_threads(2); os.nice(10)
    base = ROOT / 'output/interhuman_baseline_restore_20261016/intermask'
    cache_path = base / 'validation480.h5'
    reference_path = base / 'validation480_metrics_v2.embeddings.npz'
    old_report = json.loads((base / 'validation480_metrics_v2.json').read_text())
    output = ROOT / 'output/hhc_sota_20261016/native_baseline_val480_scoring_parity.json'
    assert not output.exists()
    data = InterHumanValidationRequestDataset()
    started = time.time()
    model, _ = load_official_interclip('cpu'); model.eval().requires_grad_(False)
    with h5py.File(cache_path, 'r') as cache:
        assert cache.attrs['status'] == 'complete' and len(cache) == 480
        assert json.loads(cache.attrs['metadata'])['request_sha256'] == data.payload['request_sha256']
        class ExistingRawCache:
            def __getitem__(self, index):
                row = data.requests[index]
                group = cache[str(index)]
                assert json.loads(group.attrs['request']) == row
                raw = group['gen'][:]
                n = row['processed_length'] // 4 * 4
                assert raw.shape == (n, 2, 262)
                normalized = (raw - data.adapter.mean) / data.adapter.std
                pair = torch.zeros(300, 2, 262)
                pair[:n] = torch.from_numpy(normalized.astype(np.float32))
                return row['identifier'], row['caption'], pair[:, 0], pair[:, 1], n
        arrays = score_main(data, ExistingRawCache(), model)
    with np.load(reference_path) as reference:
        parity = {key: {'equal': bool(np.array_equal(arrays[key], reference[key])),
            'max_abs': float(np.abs(arrays[key] - reference[key]).max())} for key in ('GT', 'text', 'raw')}
    assert all(parity[key]['equal'] for key in ('GT', 'text'))
    assert parity['raw']['max_abs'] < 1e-5
    metrics = main_metrics_from_embeddings(arrays['text'], arrays['raw'], arrays['GT'],
        load_official_metrics_module(), batch_size=96, random_seed=20261006)
    differences = {key: float(metrics[key] - old_report['metrics'][key])
                   for key in ('fid', 'matching', 'top1', 'top2', 'top3', 'diversity', 'diversity_gt')}
    assert max(abs(value) for value in differences.values()) < 1e-4
    result = {'status': 'passed_real_InterMask480_CPU_scoring_replay', 'split': 'val', 'samples': 480,
        'source_cache_sha256': file_sha(cache_path), 'reference_embeddings_sha256': file_sha(reference_path),
        'new_scorer_sha256': file_sha(ROOT / 'tools/score_interhuman_native_baselines_cpu.py'),
        'embeddings_parity': parity, 'metric_deltas_from_accepted_val': differences, 'metrics': metrics,
        'GPU_work': False, 'new_generation': False, 'formal_test_scoring': False,
        'scope': 'Same official CPU embedding/metric semantics; raw->normalized->raw float32 roundtrip only. Not new model performance or a test result.',
        'elapsed_s': time.time() - started}
    with output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
