import hashlib
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

from dataset.interhuman_generated_h5 import (
    MEAN_SHA256,
    STD_SHA256,
    InterHumanGeneratedH5Dataset,
)


def _fixture(root: Path):
    requests = [
        {'identifier': 'a', 'caption': 'first', 'processed_length': 20},
        {'identifier': 'b', 'caption': 'second', 'processed_length': 30},
    ]
    compact = json.dumps(
        requests, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode()
    content_hash = hashlib.sha256(compact).hexdigest()
    request = root / 'request.json'
    request.write_text(json.dumps({'requests': requests, 'request_sha256': content_hash}))
    request_hash = hashlib.sha256(request.read_bytes()).hexdigest()
    generation = root / 'generation.h5'
    strings = h5py.string_dtype('utf-8')
    with h5py.File(generation, 'w') as handle:
        handle.attrs['schema'] = 'interhuman_fair_generation_v1'
        handle.attrs['request_file_sha256'] = request_hash
        handle.attrs['request_content_sha256'] = content_hash
        handle.attrs['normalization_mean_sha256'] = MEAN_SHA256
        handle.attrs['normalization_std_sha256'] = STD_SHA256
        handle.attrs['sampler'] = 'sample_intra_cas'
        handle.create_dataset('names', data=np.asarray(['a', 'b'], dtype=object), dtype=strings)
        handle.create_dataset('texts', data=np.asarray(['first', 'second'], dtype=object), dtype=strings)
        handle.create_dataset('lengths', data=np.asarray([20, 30], dtype=np.int64))
        handle.create_dataset('motion1', data=np.zeros((2, 300, 262), dtype=np.float32))
        handle.create_dataset('motion2', data=np.ones((2, 300, 262), dtype=np.float32))
    return request, generation, requests


def test_generated_h5_validates_and_yields_official_tuple():
    with tempfile.TemporaryDirectory() as temporary:
        request, generation, _ = _fixture(Path(temporary))
        dataset = InterHumanGeneratedH5Dataset(
            generation, request, formal=False
        )
        assert len(dataset) == 2
        name, text, motion1, motion2, length = dataset[1]
        assert (name, text, length) == ('b', 'second', 30)
        assert motion1.shape == motion2.shape == (300, 262)
        assert motion1.sum() == 0
        assert motion2.sum() == 300 * 262
        dataset.close()


def test_generated_h5_rejects_request_order_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        request, generation, _ = _fixture(Path(temporary))
        with h5py.File(generation, 'r+') as handle:
            handle['names'][0] = 'wrong'
        try:
            InterHumanGeneratedH5Dataset(generation, request, formal=False)
        except ValueError as exc:
            assert 'names/order' in str(exc)
        else:
            raise AssertionError('request order mismatch was accepted')


def test_generated_h5_rejects_normalization_or_request_hash_mismatch():
    for attribute in ('normalization_mean_sha256', 'request_content_sha256'):
        with tempfile.TemporaryDirectory() as temporary:
            request, generation, _ = _fixture(Path(temporary))
            with h5py.File(generation, 'r+') as handle:
                handle.attrs[attribute] = 'wrong'
            try:
                InterHumanGeneratedH5Dataset(generation, request, formal=False)
            except ValueError as exc:
                assert attribute in str(exc)
            else:
                raise AssertionError(f'{attribute} mismatch was accepted')

