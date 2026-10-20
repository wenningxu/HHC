"""Request-bound public-baseline generations with method-native output lengths."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset.interhuman_generated_h5 import MEAN_SHA256, STD_SHA256, mm_repeat_seed, sha256_file

METHODS = ('intergen', 'intermask', 'timotion')


def native_length(method, length):
    if method not in METHODS:
        raise ValueError('Unsupported public baseline: ' + method)
    return int(length) // 4 * 4 if method == 'intermask' else int(length)


class NativeBaselineH5:
    """Keep the frozen GT length and actual generated length separate."""
    def __init__(self, path, request_path, entry, method, mode):
        if method not in METHODS or mode not in ('main', 'mm'):
            raise ValueError('Unsupported method/mode')
        self.path = Path(path)
        self.request_path = Path(request_path)
        self.payload = json.loads(self.request_path.read_text())
        self.requests = self.payload['requests']
        count = 1056 if mode == 'main' else 100
        self.repeats = 1 if mode == 'main' else 30
        self.mode = mode
        if len(self.requests) != count or sha256_file(self.request_path) != entry['sha256']:
            raise ValueError('Wrong frozen request file/count')
        if self.payload['request_sha256'] != entry['request_sha256']:
            raise ValueError('Wrong frozen request digest')
        with h5py.File(self.path, 'r') as handle:
            required = {'names', 'texts', 'requested_lengths', 'lengths', 'repeat_seeds', 'motion1', 'motion2'}
            if not required.issubset(handle):
                raise ValueError('Incomplete native baseline H5')
            expected_attrs = {'schema': 'interhuman_native_baseline_generation_v1', 'status': 'complete',
                'split': 'test', 'method': method, 'mode': mode,
                'request_file_sha256': entry['sha256'], 'request_content_sha256': entry['request_sha256'],
                'normalization_mean_sha256': MEAN_SHA256, 'normalization_std_sha256': STD_SHA256}
            if any(handle.attrs.get(key) != value for key, value in expected_attrs.items()):
                raise ValueError('Native baseline H5 identity/protocol mismatch')
            shape = (count, 300, 262) if mode == 'main' else (count, 30, 300, 262)
            expected_shapes = {'names': (count,), 'texts': (count,), 'requested_lengths': (count,),
                'lengths': (count,), 'repeat_seeds': (count, self.repeats), 'motion1': shape, 'motion2': shape}
            if any(handle[key].shape != value for key, value in expected_shapes.items()):
                raise ValueError('Wrong native baseline H5 shapes')
            if any(handle[key].dtype != np.float32 for key in ('motion1', 'motion2')):
                raise ValueError('Motions must be normalized float32')
            decode = lambda values: [v.decode() if isinstance(v, bytes) else str(v) for v in values]
            if decode(handle['names'][:]) != [row['identifier'] for row in self.requests]:
                raise ValueError('Names/order differ from frozen requests')
            if decode(handle['texts'][:]) != [row['caption'] for row in self.requests]:
                raise ValueError('Texts/order differ from frozen requests')
            requested = np.asarray([row['processed_length'] for row in self.requests], np.int64)
            expected = np.asarray([native_length(method, n) for n in requested], np.int64)
            if not np.array_equal(handle['requested_lengths'][:], requested) or not np.array_equal(handle['lengths'][:], expected):
                raise ValueError('Method-native or GT lengths changed')
            seeds = np.asarray([[mm_repeat_seed(row['sample_seed'], repeat) for repeat in range(self.repeats)]
                                for row in self.requests], dtype=np.uint64)
            if not np.array_equal(handle['repeat_seeds'][:], seeds):
                raise ValueError('Motion noise seeds changed')
            self.metadata = json.loads(handle.attrs['metadata'])
            if self.metadata['method'] != method or self.metadata['gt_motion_or_shape_used'] is not False:
                raise ValueError('Baseline must generate from text/length only')
        self._handle = None

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = h5py.File(self.path, 'r')
        row = self.requests[index]
        n = int(self._handle['lengths'][index])
        return (row['identifier'], row['caption'],
                torch.from_numpy(self._handle['motion1'][index]),
                torch.from_numpy(self._handle['motion2'][index]),
                n if self.mode == 'main' else torch.full((30,), n, dtype=torch.long))

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        if hasattr(self, '_handle'):
            self.close()
