"""Small runtime checks for the native-SMPL Inter-X raw contract."""
from argparse import Namespace
from pathlib import Path

import h5py
import io
import numpy as np
import torch
import torch.nn.functional as F

from dataset.interhuman_smpl_pair import body192_from_motion, model_view, pack_raw
from dataset.interhuman_interx_raw import InterHumanInterXRaw
from models.vqvae_interhuman_smpl_raw import NativeSMPLFK, RawLoss, VQSMPLRaw
from tools.reconstruct_interhuman_raw_body import load_model
from utils.losses import ReConsLossInter
from utils.quantizer_state import quantizer_training_state, restore_quantizer_training_state

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/interhuman_smpl_interx_layout_v1_20261004'


def _args(**overrides):
    values = dict(dataname='hhi', motion_rep='raw', nb_code=8, code_dim=256,
                  mu=.99, loss_explicit=3., loss_root=.1, loss_vel=100.,
                  loss_geo=.01, loss_fc=10., loss_distance_map=.1)
    values.update(overrides)
    return Namespace(**values)


def test_raw_coordinates_fk_and_dataset():
    fk = NativeSMPLFK()
    with h5py.File(DATA / 'motions.h5', 'r') as archive:
        source = archive['7778']
        motion = torch.from_numpy(source['motion30'][:12])
        beta = torch.from_numpy(source['betas'][:]).float()
    restored = fk.to_archive(fk.from_archive(motion, beta), beta)
    torch.testing.assert_close(restored[:, :22], motion[:, :22], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(restored[:, 22, [0, 1, 2, 6, 7, 8]],
                               motion[:, 22, [0, 1, 2, 6, 7, 8]], atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(restored[:, 22, [3, 4, 5, 9, 10, 11]]) == 0

    # Compare transformed native FK against the established body192 conversion.
    fk.eval()
    converted = fk.from_archive(motion, beta)
    with torch.no_grad():
        joints = fk(converted[None], beta[None])
        old = body192_from_motion(motion, beta.reshape(2, 10), load_model())
    basis = fk.basis.cpu()
    for person, actual in enumerate(joints):
        source_axes = actual[0].cpu() @ basis
        torch.testing.assert_close(source_axes, old[person, :12, :66].reshape(12, 22, 3),
                                   atol=2e-5, rtol=2e-5)

    dataset = InterHumanInterXRaw(DATA, 'val', fk, window=8,
                                  stats=(np.zeros(276, np.float32), np.ones(276, np.float32)))
    assert dataset.ids
    sample, shape, group = dataset[0]
    assert sample.shape == (8, 276) and shape.shape == (20,) and group.shape == (2,)
    assert torch.isfinite(sample).all()


def test_small_vq_encode_decode_shape_and_full():
    torch.manual_seed(17)
    model = VQSMPLRaw(_args()).eval()
    x = torch.randn(2, 8, 276)
    beta = torch.randn(2, 20)
    gender = torch.tensor([[0, 1], [1, 0]])
    with torch.no_grad():
        model.train()(x, beta, gender, type='full')  # initialize the EMA codebooks
        model.eval()
        full, commitment, perplexity = model(x, beta, gender, type='full')
        codes = model(x, beta, gender, type='encode')
        decoded = model(codes, beta, gender, type='decode')
        changed_shape = model(codes, beta + .5, gender, type='decode')
    assert codes.shape == (2, 2, 2)
    assert decoded.shape == full.shape == x.shape
    assert commitment.ndim == 0 and torch.isfinite(commitment)
    assert perplexity.shape == (2,) and torch.isfinite(perplexity).all()
    torch.testing.assert_close(full, decoded, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(decoded, changed_shape)

    # Keep the checkpoint round trip in memory; the architecture is deliberately large.
    state = quantizer_training_state(model)
    stream = io.BytesIO()
    torch.save({'model': model.state_dict(), 'quantizer': state}, stream)
    stream.seek(0)
    restored = VQSMPLRaw(_args()).eval()
    checkpoint = torch.load(stream, map_location='cpu', weights_only=False)
    restored.load_state_dict(checkpoint['model'])
    restore_quantizer_training_state(restored, checkpoint['quantizer'])
    with torch.no_grad():
        torch.testing.assert_close(restored(x, beta, gender, type='full')[0], full,
                                   atol=1e-6, rtol=1e-6)


def test_raw_loss_finite_grad_and_shared_legacy_equations():
    torch.manual_seed(23)
    device = torch.device('cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cpu')
    fk = NativeSMPLFK().to(device)
    loss_fn = RawLoss(_args(), fk, type('Identity', (), {
        'backward': staticmethod(lambda x: x),
    })()).to(device)
    target = torch.randn(1, 8, 276, device=device) * .02
    target[..., 264:267] = 0
    target[..., 270:273] = 0
    pred = (target + torch.randn_like(target) * .01).requires_grad_()
    beta = torch.zeros(1, 20, device=device)
    values = loss_fn(pred, target, beta)
    total = sum(values.values())
    assert all(torch.isfinite(value) for value in values.values())
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().max() > 0

    # RawLoss inherits these equations unchanged from the legacy Inter-X loss.
    base = ReConsLossInter(_args(), 'l1_smooth', 22, fk, normalizer=type('Identity', (), {
        'backward': staticmethod(lambda x: x),
    })()).to(device)
    base.motion_pred_de = loss_fn.motion_pred_de
    base.motion_gt_de = loss_fn.motion_gt_de
    base.motion_pred_de_joints_p1, base.motion_pred_de_joints_p2 = loss_fn.motion_pred_de_joints_p1, loss_fn.motion_pred_de_joints_p2
    base.motion_gt_de_joints_p1, base.motion_gt_de_joints_p2 = loss_fn.motion_gt_de_joints_p1, loss_fn.motion_gt_de_joints_p2
    for name in ('forward_root', 'forward_vel', 'forward_distance_map'):
        assert getattr(RawLoss, name) is getattr(ReConsLossInter, name)
        torch.testing.assert_close(getattr(loss_fn, name)(), getattr(base, name)())
    expected_distance = torch.cdist(loss_fn.motion_pred_de_joints_p1.contiguous(),
                                    loss_fn.motion_pred_de_joints_p2.contiguous())
    assert torch.isfinite(F.smooth_l1_loss(expected_distance, expected_distance))


if __name__ == '__main__':
    test_raw_coordinates_fk_and_dataset()
    test_small_vq_encode_decode_shape_and_full()
    test_raw_loss_finite_grad_and_shared_legacy_equations()
    print('PASS: Inter-X raw coordinates, FK, dataset, VQ shapes/conditioning, and loss gradients')
