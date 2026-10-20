"""CPU real-request/synthetic-motion tests for native-length main/MM I/O."""
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from dataset.interhuman_native_baseline_h5 import NativeBaselineH5, METHODS, native_length
from tools.export_interhuman_native_baselines_formal_cloud import write_native_h5, request_rng
from tools.export_hhc_text_only_formal_cloud import load_requests
from tools.hhc_text_only_formal_contract import file_sha, validate_manifests


def main():
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    passed = []
    manifest = ROOT / 'output/interhuman_fair_requests_v3/manifest.json'
    mm_manifest = ROOT / 'output/interhuman_fair_mm_requests/manifest.json'
    validate_manifests(manifest, mm_manifest)
    stub = {'main_manifest': {'path': str(manifest)}, 'mm_manifest': {'path': str(mm_manifest)}}
    # All protocol files are checked without reading GT motion or generating test outputs.
    counts = {}
    for mode in ('main', 'mm'):
        for replication in range(20):
            rows, _, _ = load_requests(stub, mode, replication)
            counts[mode] = counts.get(mode, 0) + len(rows)
    assert counts == {'main': 21120, 'mm': 2000}
    passed.append('all40_frozen_request_files_hash_counts_unique_ids_and_lengths')
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    draws = {}
    for seed in (2**63 + 17, 1234, 42):
        with request_rng(seed, 'cpu'):
            draws[seed] = (random.random(), np.random.random(), torch.rand(4))
    for seed in reversed(list(draws)):
        with request_rng(seed, 'cpu'):
            assert random.random() == draws[seed][0] and np.random.random() == draws[seed][1]
            assert torch.equal(torch.rand(4), draws[seed][2])
    assert random.getstate() == state[0] and np.array_equal(np.random.get_state()[1], state[1][1])
    assert torch.equal(torch.get_rng_state(), state[2])
    passed.append('per_request_uint64_RNG_order_invariant_and_global_states_preserved')

    def rejects(name, action):
        try: action()
        except (ValueError, FileExistsError): passed.append(name)
        else: raise AssertionError('Invalid native H5 accepted: ' + name)

    summary = {}
    with tempfile.TemporaryDirectory(prefix='native_baseline_cpu_', dir=ROOT / 'output') as directory:
        directory = Path(directory)
        for method in METHODS:
            for mode in ('main', 'mm'):
                rows, request_path, entry = load_requests(stub, mode, 0)
                calls = []
                def generate(row, seed):
                    n = native_length(method, row['processed_length'])
                    calls.append((row['identifier'], seed))
                    return torch.full((n, 2, 262), (seed % 997) / 997, dtype=torch.float32)
                metadata = {'method': method, 'gt_motion_or_shape_used': False,
                            'scope': 'CPU synthetic motion protocol test, not model inference'}
                path = directory / (method + '_' + mode + '.h5')
                with patch('builtins.print'):
                    write_native_h5(path, rows, request_path, entry, method, mode, generate, metadata)
                data = NativeBaselineH5(path, request_path, entry, method, mode)
                assert len(data) == len(rows)
                repeats = 1 if mode == 'main' else 30
                from dataset.interhuman_generated_h5 import mm_repeat_seed
                assert calls == [(row['identifier'], mm_repeat_seed(row['sample_seed'], repeat))
                                 for repeat in range(repeats) for row in rows]
                for i in range(len(rows)):
                    _, text, a, b, length = data[i]
                    n = native_length(method, rows[i]['processed_length'])
                    assert text == rows[i]['caption']
                    assert int(length if mode == 'main' else length[0]) == n
                    assert torch.isfinite(a).all() and torch.isfinite(b).all()
                    assert not torch.count_nonzero(a[n:] if mode == 'main' else a[:, n:])
                    assert not torch.count_nonzero(b[n:] if mode == 'main' else b[:, n:])
                data.close()
                label = method + '_' + mode
                passed.append(label + '_native_lengths_order_padding_seed_roundtrip')
                summary[label] = {'prompts': len(rows), 'repeats': repeats,
                    'length_shortening_frames_mean': float(np.mean([r['processed_length'] - native_length(method, r['processed_length']) for r in rows])),
                    'bytes': path.stat().st_size}
                rejects(label + '_replay_rejected', lambda: write_native_h5(path, rows, request_path, entry, method, mode, generate, metadata))
                with h5py.File(path, 'r+') as h:
                    original = int(h['lengths'][0]); h['lengths'][0] = original - 1
                rejects(label + '_wrong_native_length_rejected', lambda: NativeBaselineH5(path, request_path, entry, method, mode))
                with h5py.File(path, 'r+') as h:
                    h['lengths'][0] = original; old_seed = h['repeat_seeds'][0, 0]; h['repeat_seeds'][0, 0] = 0
                rejects(label + '_wrong_seed_rejected', lambda: NativeBaselineH5(path, request_path, entry, method, mode))
                with h5py.File(path, 'r+') as h:
                    h['repeat_seeds'][0, 0] = old_seed
                    changed = dict(metadata, gt_motion_or_shape_used=True); h.attrs['metadata'] = json.dumps(changed)
                rejects(label + '_GT_condition_rejected', lambda: NativeBaselineH5(path, request_path, entry, method, mode))
    config = json.loads((ROOT / 'migration/hhc-train-shape-validation-v2.json').read_text())
    assert all(file_sha(ROOT / name) == digest for name, digest in config['sources'].items())
    passed.append('29_active_training_validation_sources_unchanged')
    result = {'status': 'passed_CPU_native_baseline_protocol', 'checks': passed, 'request_counts': counts,
              'synthetic_H5_tests': summary, 'all_temporary_synthetic_H5_deleted': True,
              'GPU_work': False, 'real_baseline_generation': False,
              'scope': 'Native-length I/O and RNG acceptance only; real model inference and formal scores remain required',
              'sources': {p: file_sha(ROOT / p) for p in ('dataset/interhuman_native_baseline_h5.py',
                  'tools/export_interhuman_native_baselines_formal_cloud.py',
                  'tools/score_interhuman_native_baselines_cpu.py', 'tools/audit_interhuman_native_baseline_protocol_cpu.py')}}
    output = ROOT / 'output/hhc_sota_20261016/native_baseline_protocol_cpu_audit.json'
    with output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
