#!/usr/bin/env python3
"""CPU-only matched-sample audit against the pinned official dataset.__getitem__."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dataset.interhuman_root_rotation import RootRotationData


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.limit > 0 and not args.output.exists()
    torch.set_num_threads(1)
    try:
        os.nice(10)
    except AttributeError:
        pass
    data = RootRotationData('train')
    source = ROOT/'third_party/intergen_base_20261008'
    # RootRotationData loads data.*, whereas official InterGen uses utils.*.
    # Refuse module shadowing and report the actual imported official paths.
    assert not any(k == 'utils' or k.startswith('utils.') for k in sys.modules)
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('official_intergen_dataset_audit', source/'datasets/interhuman.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    origins = {key: str(Path(value.__file__).resolve()) for key, value in sys.modules.items()
               if (key == 'utils' or key.startswith('utils.')) and getattr(value, '__file__', None)}
    assert all(str(source) in path for path in origins.values()), origins
    # Select deterministic IDs spanning train split order, without consulting validation/test.
    indices = np.linspace(0, len(data.ids)-1, min(args.limit, len(data.ids)), dtype=int)
    ids = [data.ids[i] for i in indices]
    cases = []
    for identifier in ids:
        paths = [data.adapter._person_path(identifier, person) for person in (1, 2)]
        raw_hashes = [sha(path) for path in paths]
        length = data.adapter._pair_raw_length(identifier)
        starts = sorted({0, (length-301)//2}) if length > 301 else [0]
        for mirror in (False, True):
            loaded = [official.load_motion(str(path), 15, swap=mirror) for path in paths]
            motions = [pair[1 if mirror else 0] for pair in loaded]
            for start in starts:
                for anchor in (1, 2):
                    for swap in (False, True):
                        # Exercise the unmodified official __getitem__, avoiding full dataset init/cache.
                        item = official.InterHumanDataset.__new__(official.InterHumanDataset)
                        item.cache = True
                        item.max_length = item.max_gt_length = 300
                        item.data_list = [{'name': identifier, 'motion_id': 0, 'swap': mirror, 'texts': ['audit']}]
                        item.motion_dict = {0: motions}
                        def choose(values):
                            return 'audit' if isinstance(values[0], str) else start
                        with patch.object(random, 'choice', side_effect=choose), patch.object(np.random, 'rand', side_effect=[.75 if anchor == 2 else .25, .75 if swap else .25]):
                            _, _, left, right, n = item[0]
                        expected = np.stack([left[:n], right[:n]], 1)
                        actual, _ = data.pair(identifier, start, mirror, anchor, swap)
                        actual = actual[..., :262]
                        assert actual.shape == expected.shape
                        delta = np.abs(actual.astype(np.float64)-expected)
                        # Official padding promotes arrays to float64; compare after the trainer's FP32 cast too.
                        expected32 = expected.astype(np.float32)
                        cases.append({'id': identifier, 'mirror': mirror, 'start': start, 'anchor': anchor,
                                      'swap': swap, 'frames': n, 'raw_sha256': raw_hashes,
                                      'max_abs_raw': float(delta.max()),
                                      'bitwise_equal_float32': bool(np.array_equal(actual, expected32)),
                                      'max_abs_float32': float(np.max(np.abs(actual-expected32))),
                                      'channel_max_abs': {'position': float(delta[..., :66].max()),
                                                          'velocity': float(delta[..., 66:132].max()),
                                                          'rotation': float(delta[..., 132:258].max()),
                                                          'contact': float(delta[..., 258:262].max())}})
    # Confirm the normalizers actually read by official loss and current adapter.
    normalizers = {}
    for suffix in ('mean', 'std'):
        p = source/'data'/('global_'+suffix+'.npy')
        current = getattr(data.adapter, suffix)
        if p.exists():
            value = np.load(p)
            normalizers[suffix] = {'path': str(p), 'sha256': sha(p), 'shape': list(value.shape),
                                  'bitwise_equal_float32': bool(np.array_equal(value.astype(np.float32), current)),
                                  'max_abs': float(np.max(np.abs(value.astype(np.float64)-current)))}
        else:
            normalizers[suffix] = {'path': str(p), 'status': 'missing_not_verified'}
    assert not torch.cuda.is_initialized()
    report = {'schema': 'intergen_rotation_matched_data_cpu_v1',
              'scope': 'Matched official dataset.__getitem__ and custom first262 only; no model/GPU, no full distribution or training equivalence claim',
              'official_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
              'module_origins': origins, 'cuda_initialized': False, 'split': 'train', 'train_ids_count': len(data.ids),
              'ids': ids, 'cases_count': len(cases), 'all_float32_bitwise_equal': all(c['bitwise_equal_float32'] for c in cases),
              'worst_max_abs_raw': max(c['max_abs_raw'] for c in cases), 'normalizers': normalizers,
              'sources': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'dataset/interhuman_root_rotation.py',
                         ROOT/'dataset/interhuman_hhc_adapter.py', source/'datasets/interhuman.py', source/'utils/preprocess.py', source/'utils/utils.py']},
              'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['cases_count', 'all_float32_bitwise_equal', 'worst_max_abs_raw', 'normalizers', 'cuda_initialized']}), flush=True)


if __name__ == '__main__':
    main()
