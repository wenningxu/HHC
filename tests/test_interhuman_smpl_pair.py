"""Local regression checks for native SMPL paired packing and generated FK."""
import pickle
from pathlib import Path

import numpy as np
import torch

from dataset.interhuman_smpl_pair import (
    pack_raw, unpack_raw, model_view, refresh_velocity, body192_from_motion, to_interhuman,
)
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from tools.reconstruct_interhuman_raw_body import load_model

ROOT = Path(__file__).resolve().parents[1]


def test_archive_and_velocity():
    with (ROOT/'dataset/interhuman_official/motions/7778.pkl').open('rb') as stream:
        raw = pickle.load(stream)
    axis, beta = pack_raw(raw)
    restored = unpack_raw(axis, beta)
    for name in restored:
        for key, array in restored[name].items():
            assert array.dtype == raw[name][key].dtype
            assert array.tobytes() == raw[name][key].tobytes()
    motion, indices = model_view(axis, raw['mocap_framerate'])
    assert indices[:5].tolist() == [0, 1, 3, 5, 7]
    cropped = refresh_velocity(motion[10:20])
    batched = refresh_velocity(motion[None, 10:20])
    torch.testing.assert_close(cropped, batched[0])
    for offset in (0, 6):
        assert torch.count_nonzero(cropped[0, 22, offset+3:offset+6]) == 0
        torch.testing.assert_close(cropped[1:, 22, offset+3:offset+6],
                                   cropped[1:, 22, offset:offset+3]-cropped[:-1, 22, offset:offset+3])


def test_shape_fk_gradient_and_official_conversion():
    with (ROOT/'dataset/interhuman_official/motions/7778.pkl').open('rb') as stream:
        raw = pickle.load(stream)
    axis, beta = pack_raw(raw)
    motion, _ = model_view(axis, raw['mocap_framerate'])
    motion = motion[:12].requires_grad_()
    beta = torch.from_numpy(beta).requires_grad_()
    model = load_model()
    body = body192_from_motion(motion, beta, model)
    zero_shape = body192_from_motion(motion, torch.zeros_like(beta), model)
    assert (body[..., :66]-zero_shape[..., :66]).abs().max() > 0.001
    body.square().mean().backward()
    for value in (motion.grad, beta.grad):
        assert value is not None and torch.isfinite(value).all() and value.abs().max() > 0
    for person in range(2):
        official = np.load(ROOT/f'dataset/interhuman_official/motions_processed/person{person+1}/7778.npy')[:12]
        expected = np.concatenate([official[:, :66], official[:, 186:312]], axis=-1)
        assert np.max(np.abs(body[person].detach().numpy()-expected)) < 2e-6
    features = to_interhuman(motion, beta, model, OfficialInterHumanAdapter())
    assert features.shape == (11, 524) and np.isfinite(features).all()


if __name__ == '__main__':
    test_archive_and_velocity()
    test_shape_fk_gradient_and_official_conversion()
    print('PASS: archive, resampling, cropped/batched velocity, shape FK, gradients, official conversion')
