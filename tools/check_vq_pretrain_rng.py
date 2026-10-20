"""CPU mechanism check, not proof of the January training trajectory.

Run from HHC: python tools/check_vq_pretrain_rng.py
"""
import json
import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from models.quantize_cnn import QuantizeEMAReset


def real_first_batches():
    from dataset import dataset_interx, dataset_interx_TM
    from utils.word_vectorizer import WordVectorizer

    train = dataset_interx.DATALoader(
        'hhi', 256, 'dataset/interx/splits/train.txt',
        'dataset/interx/motions/train.h5', num_workers=0, motion_rep='raw')
    validation = dataset_interx_TM.DATALoader(
        'hhi', 32, 'dataset/interx/splits/val.txt',
        WordVectorizer('./dataset/interx/glove', 'hhi_vab'),
        'dataset/interx/motions/val.h5', num_workers=0, shuffle=False)
    batches = []
    for evaluate, isolate in ((False, False), (True, False), (True, True)):
        torch.manual_seed(123)
        state = torch.get_rng_state()
        if evaluate:
            for _ in validation:
                pass
            assert not torch.equal(state, torch.get_rng_state())
            if isolate:
                torch.set_rng_state(state)
        batches.append(next(iter(train))[0])
    assert torch.equal(batches[0], batches[2])
    assert not torch.equal(batches[0], batches[1])
    hashes = [[hashlib.sha256(sample.numpy().tobytes()).hexdigest()
               for sample in batch] for batch in batches]
    print(json.dumps({
        'status': 'PASS',
        'scope': 'real raw loader only, seed123 prefix, zero workers; no historical model/RNG reconstruction',
        'batch_size': len(batches[0]),
        'validation_batches': len(validation),
        'same_position_clips': sum(a == b for a, b in zip(hashes[0], hashes[1])),
        'shared_unique_clips': len(set(hashes[0]) & set(hashes[1])),
        'isolated_matches_baseline': True,
        'batch_sha256': [hashlib.sha256(b.numpy().tobytes()).hexdigest() for b in batches],
    }, indent=2))


def first_update(pre_eval, isolate):
    torch.manual_seed(123)
    quantizer = QuantizeEMAReset(32, 2, SimpleNamespace(mu=0.99))
    train = DataLoader(torch.arange(64), batch_size=8, shuffle=True)
    validation = DataLoader(torch.arange(16), batch_size=4, shuffle=False)
    if pre_eval:
        state = torch.get_rng_state()
        quantizer.eval()
        for batch in validation:
            quantizer(batch.float().view(-1, 1, 1).repeat(1, 2, 2))
        assert not quantizer.init
        assert torch.count_nonzero(quantizer.codebook) == 0
        assert quantizer.code_sum is None and quantizer.code_count is None
        assert not torch.equal(state, torch.get_rng_state())
        if isolate:
            torch.set_rng_state(state)
    quantizer.train()
    batch = next(iter(train))
    quantizer(batch.float().view(-1, 1, 1).repeat(1, 2, 2))
    assert quantizer.init and torch.isfinite(quantizer.codebook).all()
    return batch, quantizer.codebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-data', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.real_data:
        real_first_batches()
        sys.exit(0)
    baseline = first_update(False, False)
    unisolated = first_update(True, False)
    isolated = first_update(True, True)
    assert all(torch.equal(a, b) for a, b in zip(baseline, isolated))
    assert all(not torch.equal(a, b) for a, b in zip(baseline, unisolated))
    print(json.dumps({
        'status': 'PASS',
        'scope': 'synthetic CPU mechanism, not historical quality attribution',
        'eval_initializes_codebook': False,
        'baseline_first_batch': baseline[0].tolist(),
        'unisolated_eval_first_batch': unisolated[0].tolist(),
        'isolated_eval_matches_baseline': True,
        'unisolated_codebook_max_abs_difference': float(
            (baseline[1] - unisolated[1]).abs().max()),
    }, indent=2))
