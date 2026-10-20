"""Validated HDF5 datasets for fair InterHuman generations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


MEAN_SHA256 = '2f97b88889c773551cd597c32e88bfa070c2fb1721a992a80bfb7cf4c22e03b6'
STD_SHA256 = 'dd5988b979e9803786ab497f5945316c2d52b97c90b593d0427220fd49b03e4b'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_strings(values) -> list[str]:
    return [value.decode('utf-8') if isinstance(value, bytes) else str(value) for value in values]


def mm_repeat_seed(sample_seed: int, repeat_index: int) -> int:
    """Derive an independent, reproducible seed for one MM generation."""
    payload = f'{int(sample_seed)}:repeat:{int(repeat_index)}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


class InterHumanGeneratedH5Dataset(Dataset):
    """Expose a generated H5 in the exact official evaluator tuple format."""

    required_datasets = ('names', 'texts', 'lengths', 'motion1', 'motion2')

    def __init__(
        self,
        generation_path,
        request_path,
        request_entry=None,
        formal=True,
    ):
        self.generation_path = Path(generation_path).resolve()
        self.request_path = Path(request_path).resolve()
        self.request_payload = json.loads(self.request_path.read_text())
        self.requests = self.request_payload['requests']
        self.request_file_sha256 = sha256_file(self.request_path)
        if request_entry is not None:
            if self.request_file_sha256 != request_entry['sha256']:
                raise ValueError('request file hash differs from its manifest entry')
            if self.request_payload['request_sha256'] != request_entry['request_sha256']:
                raise ValueError('request content hash differs from its manifest entry')
        self._handle = None

        with h5py.File(self.generation_path, 'r') as handle:
            missing = sorted(set(self.required_datasets) - set(handle))
            if missing:
                raise ValueError(f'generation H5 lacks datasets {missing}')
            if handle.attrs.get('schema') != 'interhuman_fair_generation_v1':
                raise ValueError('unsupported InterHuman generation H5 schema')
            for attr, expected in (
                ('request_file_sha256', self.request_file_sha256),
                ('request_content_sha256', self.request_payload['request_sha256']),
                ('normalization_mean_sha256', MEAN_SHA256),
                ('normalization_std_sha256', STD_SHA256),
            ):
                actual = handle.attrs.get(attr)
                if actual != expected:
                    raise ValueError(f'generation H5 {attr} mismatch: {actual!r} != {expected!r}')
            count = len(self.requests)
            if formal and count != 1056:
                raise ValueError(f'formal InterHuman generation requires 1056 requests, got {count}')
            expected_shapes = {
                'names': (count,), 'texts': (count,), 'lengths': (count,),
                'motion1': (count, 300, 262), 'motion2': (count, 300, 262),
            }
            for name, shape in expected_shapes.items():
                if handle[name].shape != shape:
                    raise ValueError(f'generation H5 {name} shape {handle[name].shape} != {shape}')
            if handle['motion1'].dtype != np.float32 or handle['motion2'].dtype != np.float32:
                raise ValueError('InterHuman generated motions must be float32')
            names = _decode_strings(handle['names'][:])
            texts = _decode_strings(handle['texts'][:])
            lengths = handle['lengths'][:].astype(np.int64)
            expected_names = [row['identifier'] for row in self.requests]
            expected_texts = [row['caption'] for row in self.requests]
            expected_lengths = np.asarray(
                [row['processed_length'] for row in self.requests], dtype=np.int64
            )
            if names != expected_names:
                raise ValueError('generation H5 names/order differ from frozen requests')
            if texts != expected_texts:
                raise ValueError('generation H5 texts/order differ from frozen requests')
            if not np.array_equal(lengths, expected_lengths):
                raise ValueError('generation H5 lengths differ from frozen requests')
            if not np.all((lengths >= 14) & (lengths <= 299)):
                raise ValueError('generation H5 contains invalid processed lengths')
            self.metadata = {
                key: (
                    value.decode('utf-8') if isinstance(value, bytes) else
                    value.item() if isinstance(value, np.generic) else value
                )
                for key, value in handle.attrs.items()
            }

    def __len__(self):
        return len(self.requests)

    def _h5(self):
        if self._handle is None:
            self._handle = h5py.File(self.generation_path, 'r')
        return self._handle

    def __getitem__(self, item):
        handle = self._h5()
        return (
            _decode_strings(handle['names'][item:item + 1])[0],
            _decode_strings(handle['texts'][item:item + 1])[0],
            torch.from_numpy(handle['motion1'][item]).float(),
            torch.from_numpy(handle['motion2'][item]).float(),
            int(handle['lengths'][item]),
        )

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        if hasattr(self, '_handle'):
            self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'] = None
        return state


class InterHumanMMGeneratedH5Dataset(Dataset):
    """Expose frozen 100-prompt x 30-repeat generations to InterCLIP."""

    required_datasets = (
        'names', 'texts', 'lengths', 'repeat_seeds', 'motion1', 'motion2',
    )

    def __init__(
        self,
        generation_path,
        request_path,
        request_entry=None,
        formal=True,
    ):
        self.generation_path = Path(generation_path).resolve()
        self.request_path = Path(request_path).resolve()
        self.request_payload = json.loads(self.request_path.read_text())
        self.requests = self.request_payload['requests']
        self.request_file_sha256 = sha256_file(self.request_path)
        if request_entry is not None:
            if self.request_file_sha256 != request_entry['sha256']:
                raise ValueError('MM request file hash differs from its manifest entry')
            if self.request_payload['request_sha256'] != request_entry['request_sha256']:
                raise ValueError('MM request content hash differs from its manifest entry')
        self._handle = None

        prompt_count = len(self.requests)
        repeat_count = int(self.request_payload['mm_repeats'])
        if formal and (prompt_count, repeat_count) != (100, 30):
            raise ValueError(
                'formal InterHuman multimodality requires 100 prompts x 30 repeats, '
                f'got {prompt_count} x {repeat_count}'
            )
        with h5py.File(self.generation_path, 'r') as handle:
            missing = sorted(set(self.required_datasets) - set(handle))
            if missing:
                raise ValueError(f'MM generation H5 lacks datasets {missing}')
            if handle.attrs.get('schema') != 'interhuman_fair_mm_generation_v1':
                raise ValueError('unsupported InterHuman MM generation H5 schema')
            for attr, expected in (
                ('request_file_sha256', self.request_file_sha256),
                ('request_content_sha256', self.request_payload['request_sha256']),
                ('normalization_mean_sha256', MEAN_SHA256),
                ('normalization_std_sha256', STD_SHA256),
            ):
                actual = handle.attrs.get(attr)
                if actual != expected:
                    raise ValueError(f'MM generation H5 {attr} mismatch: {actual!r} != {expected!r}')
            expected_shapes = {
                'names': (prompt_count,),
                'texts': (prompt_count,),
                'lengths': (prompt_count,),
                'repeat_seeds': (prompt_count, repeat_count),
                'motion1': (prompt_count, repeat_count, 300, 262),
                'motion2': (prompt_count, repeat_count, 300, 262),
            }
            for name, shape in expected_shapes.items():
                if handle[name].shape != shape:
                    raise ValueError(f'MM generation H5 {name} shape {handle[name].shape} != {shape}')
            if handle['motion1'].dtype != np.float32 or handle['motion2'].dtype != np.float32:
                raise ValueError('InterHuman MM generated motions must be float32')
            names = _decode_strings(handle['names'][:])
            texts = _decode_strings(handle['texts'][:])
            lengths = handle['lengths'][:].astype(np.int64)
            repeat_seeds = handle['repeat_seeds'][:].astype(np.uint64)
            if names != [row['identifier'] for row in self.requests]:
                raise ValueError('MM generation H5 names/order differ from frozen requests')
            if texts != [row['caption'] for row in self.requests]:
                raise ValueError('MM generation H5 texts/order differ from frozen requests')
            expected_lengths = np.asarray(
                [row['processed_length'] for row in self.requests], dtype=np.int64
            )
            if not np.array_equal(lengths, expected_lengths):
                raise ValueError('MM generation H5 lengths differ from frozen requests')
            expected_repeat_seeds = np.asarray([
                [mm_repeat_seed(row['sample_seed'], repeat) for repeat in range(repeat_count)]
                for row in self.requests
            ], dtype=np.uint64)
            if not np.array_equal(repeat_seeds, expected_repeat_seeds):
                raise ValueError('MM generation H5 repeat seeds differ from frozen derivation')
            if not np.all((lengths >= 14) & (lengths <= 299)):
                raise ValueError('MM generation H5 contains invalid processed lengths')
            self.metadata = {
                key: (
                    value.decode('utf-8') if isinstance(value, bytes) else
                    value.item() if isinstance(value, np.generic) else value
                )
                for key, value in handle.attrs.items()
            }
        self.repeat_count = repeat_count

    def __len__(self):
        return len(self.requests)

    def _h5(self):
        if self._handle is None:
            self._handle = h5py.File(self.generation_path, 'r')
        return self._handle

    def __getitem__(self, item):
        handle = self._h5()
        length = int(handle['lengths'][item])
        return (
            _decode_strings(handle['names'][item:item + 1])[0],
            _decode_strings(handle['texts'][item:item + 1])[0],
            torch.from_numpy(handle['motion1'][item]).float(),
            torch.from_numpy(handle['motion2'][item]).float(),
            torch.full((self.repeat_count,), length, dtype=torch.long),
        )

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        if hasattr(self, '_handle'):
            self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'] = None
        return state
