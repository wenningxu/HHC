import hashlib
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

from dataset.interhuman_generated_h5 import (
    MEAN_SHA256,
    STD_SHA256,
    InterHumanMMGeneratedH5Dataset,
    mm_repeat_seed,
)


def _fixture(root: Path):
    requests = [
        {'identifier': 'a', 'caption': 'first', 'processed_length': 20, 'sample_seed': 4},
        {'identifier': 'b', 'caption': 'second', 'processed_length': 30, 'sample_seed': 9},
    ]
    compact = json.dumps(
        requests, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode()
    content_hash = hashlib.sha256(compact).hexdigest()
    request = root / 'request.json'
    request.write_text(json.dumps({
        'requests': requests, 'request_sha256': content_hash, 'mm_repeats': 3,
    }))
    request_hash = hashlib.sha256(request.read_bytes()).hexdigest()
    generation = root / 'generation.h5'
    strings = h5py.string_dtype('utf-8')
    repeat_seeds = np.asarray([
        [mm_repeat_seed(row['sample_seed'], repeat) for repeat in range(3)]
        for row in requests
    ], dtype=np.uint64)
    with h5py.File(generation, 'w') as handle:
        handle.attrs['schema'] = 'interhuman_fair_mm_generation_v1'
        handle.attrs['request_file_sha256'] = request_hash
        handle.attrs['request_content_sha256'] = content_hash
        handle.attrs['normalization_mean_sha256'] = MEAN_SHA256
        handle.attrs['normalization_std_sha256'] = STD_SHA256
        handle.create_dataset('names', data=np.asarray(['a', 'b'], dtype=object), dtype=strings)
        handle.create_dataset('texts', data=np.asarray(['first', 'second'], dtype=object), dtype=strings)
        handle.create_dataset('lengths', data=np.asarray([20, 30], dtype=np.int64))
        handle.create_dataset('repeat_seeds', data=repeat_seeds)
        handle.create_dataset('motion1', data=np.zeros((2, 3, 300, 262), dtype=np.float32))
        handle.create_dataset('motion2', data=np.ones((2, 3, 300, 262), dtype=np.float32))
    return request, generation


def test_mm_generated_h5_validates_and_yields_prompt_repeats():
    with tempfile.TemporaryDirectory() as temporary:
        request, generation = _fixture(Path(temporary))
        dataset = InterHumanMMGeneratedH5Dataset(generation, request, formal=False)
        name, text, motion1, motion2, lengths = dataset[1]
        assert (name, text) == ('b', 'second')
        assert motion1.shape == motion2.shape == (3, 300, 262)
        assert lengths.tolist() == [30, 30, 30]
        dataset.close()


def test_mm_generated_h5_rejects_repeat_seed_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        request, generation = _fixture(Path(temporary))
        with h5py.File(generation, 'r+') as handle:
            handle['repeat_seeds'][0, 1] = 123
        try:
            InterHumanMMGeneratedH5Dataset(generation, request, formal=False)
        except ValueError as exc:
            assert 'repeat seeds' in str(exc)
        else:
            raise AssertionError('MM repeat seed mismatch was accepted')
