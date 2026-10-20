import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from utils.interhuman_vq_loss import InterHumanPairVQLoss, InterHumanVQLoss


INTERMASK = Path('/mnt/8T-HDD/wenning/InterMask')


def test_interhuman_geometric_terms_match_released_intermask():
    adapter = OfficialInterHumanAdapter()
    target_np, _ = adapter.process_pair('3292', normalize=True)
    target_np = target_np[:32][None]
    rng = np.random.default_rng(41)
    prediction_np = target_np + rng.normal(0, 0.015, target_np.shape).astype(np.float32)

    loss = InterHumanVQLoss()
    ours = loss(
        torch.from_numpy(target_np), torch.from_numpy(prediction_np), torch.tensor(0.0)
    )
    names = (
        'reconstruction', 'explicit_position', 'position_velocity',
        'bone_length', 'rotation_geodesic', 'foot_sliding',
    )

    with tempfile.TemporaryDirectory() as temporary:
        input_path = Path(temporary) / 'motions.npz'
        np.savez(input_path, target=target_np, prediction=prediction_np)
        source = r'''
import json, numpy as np, sys, torch
from models.losses import Geometric_Losses
d=np.load(sys.argv[1])
target=torch.from_numpy(d['target']).float()
prediction=torch.from_numpy(d['prediction']).float()
values=Geometric_Losses('l1_smooth',22,'interhuman','cpu').forward(target,prediction)[:6]
print(json.dumps([float(value) for value in values]))
'''
        completed = subprocess.run(
            [sys.executable, '-c', source, str(input_path)], cwd=INTERMASK,
            check=True, text=True, capture_output=True,
        )
        released = json.loads(completed.stdout.strip().splitlines()[-1])

    for name, expected in zip(names, released):
        torch.testing.assert_close(
            ours[name], torch.tensor(expected), rtol=2e-6, atol=2e-7,
            msg=lambda message, name=name: f'{name}: {message}',
        )


def test_total_uses_released_default_weights_and_commitment():
    target = torch.zeros(1, 4, 262)
    prediction = target.clone()
    # Valid identity-like 6D rotations avoid zero-vector normalization NaNs.
    target[..., 132:258] = torch.tensor([1, 0, 0, 0, 1, 0]).repeat(21)
    prediction[..., 132:258] = target[..., 132:258]
    result = InterHumanVQLoss()(target, prediction, torch.tensor(2.0))
    expected = result['reconstruction']
    weights = {
        'commitment': 0.02, 'explicit_position': 1.0,
        'position_velocity': 100.0, 'bone_length': 5.0,
        'rotation_geodesic': 0.01, 'foot_sliding': 500.0,
    }
    for name, weight in weights.items():
        expected = expected + weight * result[name]
    torch.testing.assert_close(result['total'], expected)


def test_pair_loss_is_symmetric_mean_and_counts_commitment_once():
    generator = torch.Generator().manual_seed(52)
    person1 = torch.randn(2, 8, 262, generator=generator)
    person2 = torch.randn(2, 8, 262, generator=generator)
    pred1 = person1 + 0.01 * torch.randn(person1.shape, generator=generator)
    pred2 = person2 + 0.02 * torch.randn(person2.shape, generator=generator)
    # Avoid invalid zero-norm 6D rotations (random values are almost surely valid).
    commitment = torch.tensor(1.25)
    single = InterHumanVQLoss()
    first = single(person1, pred1, torch.tensor(0.0))
    second = single(person2, pred2, torch.tensor(0.0))
    paired = InterHumanPairVQLoss()(
        torch.cat([person1, person2], dim=-1),
        torch.cat([pred1, pred2], dim=-1), commitment,
    )
    for name in (
        'reconstruction', 'explicit_position', 'position_velocity',
        'bone_length', 'rotation_geodesic', 'foot_sliding',
    ):
        torch.testing.assert_close(paired[name], (first[name] + second[name]) / 2)
    assert paired['commitment'] is commitment
    expected_total = paired['reconstruction'] + 0.02 * commitment
    for name, weight in InterHumanPairVQLoss().weights.items():
        if name != 'commitment':
            expected_total = expected_total + weight * paired[name]
    torch.testing.assert_close(paired['total'], expected_total)
