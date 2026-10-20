"""Official 262D features plus aligned, column-interleaved root rotation (6D)."""
from functools import lru_cache
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from smplx.lbs import batch_rodrigues
from torch.utils.data import Dataset

from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from tools.reconstruct_interhuman_raw_body import load_model

ROOT = Path(__file__).resolve().parents[1]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
MIRROR = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20])


def rotation_matrix(six):
    """InterHuman columns are interleaved, not HHC's two consecutive rows."""
    cols = six.reshape(*six.shape[:-1], 3, 2)
    a = torch.nn.functional.normalize(cols[..., 0], dim=-1, eps=1e-6)
    b = cols[..., 1] - (a*cols[..., 1]).sum(-1, keepdim=True)*a
    b = torch.nn.functional.normalize(b, dim=-1, eps=1e-6)
    return torch.stack((a, b, torch.linalg.cross(a, b)), -1)


def fk_positions(raw, offsets):
    """[B,T,2,268], [B,2,22,3] -> joint positions. Root position is pelvis."""
    local = rotation_matrix(raw[..., 132:258].reshape(*raw.shape[:-1], 21, 6))
    world = [rotation_matrix(raw[..., 262:268])]
    positions = [raw[..., :3]]
    for j in range(1, 22):
        parent = PARENTS[j]
        positions.append(positions[parent] + (world[parent] @ offsets[:, None, :, j, :, None]).squeeze(-1))
        world.append(world[parent] @ local[..., j-1, :, :])
    return torch.stack(positions, -2)


class RootRotationData(Dataset):
    def __init__(self, split='train'):
        self.adapter = OfficialInterHumanAdapter(data_root=ROOT/'dataset/interhuman_official')
        self.ids = self.adapter.split_ids(split, require_text=True, min_raw_length=15)
        self.split = split
        model = load_model()
        with torch.no_grad():
            self.rest0 = (model.J_regressor @ model.v_template)[:22].numpy()
            self.restshape = torch.einsum('jv,vck->jck', model.J_regressor, model.shapedirs)[:22].numpy()

    @lru_cache(maxsize=32)
    def source(self, identifier):
        motions = self.adapter._load_raw_pair(identifier, False, 15)
        with (self.adapter.data_root/'motions'/f'{identifier}.pkl').open('rb') as handle:
            raw = pickle.load(handle)
        roots, rests = [], []
        for p, name in enumerate(('person1', 'person2')):
            person = raw[name]
            index = np.arange(0, len(person['trans']), float(raw['mocap_framerate'])/30).astype(np.int64)
            assert len(index) == len(motions[p])
            roots.append(batch_rodrigues(torch.from_numpy(person['root_orient'][index]).float()).numpy())
            rests.append(self.rest0 + np.einsum('jck,k->jc', self.restshape, person['betas']))
        return motions, roots, rests

    def pair(self, identifier, start=0, mirror=False, anchor=1, swap=False, max_frames=300):
        motions, roots, rests = self.source(identifier)
        stop = min(start+max_frames, len(motions[0]))
        assert 0 <= start < stop-1 and anchor in (1, 2)
        motions = [x[start:stop].copy() for x in motions]
        roots = [x[start:stop].copy() for x in roots]
        rests = [x.copy() for x in rests]
        reflection = np.diag([-1., 1., 1.]).astype(np.float32)
        if mirror:
            motions = [self.adapter.data_utils.swap_left_right(x, 22) for x in motions]
            roots = [reflection @ x @ reflection for x in roots]
            rests = [x[MIRROR] @ reflection for x in rests]
        # Both people end in the anchor's orientation, even though each initially
        # undergoes process_motion_np and the second then a relative transform.
        _, q, _ = self.adapter.interhuman.process_motion_np(motions[anchor-1].copy(), .001, 0, 22)
        qmatrix = self.adapter.quaternion.qrot_np(np.broadcast_to(q, (3, 4)).copy(), np.eye(3)).T
        transform = qmatrix @ self.adapter.data_utils.trans_matrix.numpy()
        roots = [(transform @ x)[:-1, :, :2].reshape(stop-start-1, 6) for x in roots]
        people = self.adapter.process_loaded_pair(*motions, anchor_person=anchor, output_person_swap=swap, normalize=False)
        order = [0, 1] if anchor == 1 else [1, 0]
        if swap:
            order.reverse()
        features = np.stack([np.concatenate((people[i], roots[p]), -1) for i, p in enumerate(order)], 1).astype(np.float32)
        offsets = np.stack([rests[p] for p in order]).astype(np.float32)
        offsets[:, 1:] -= offsets[:, PARENTS[1:]]
        offsets[:, 0] = 0
        return features, offsets

    def __len__(self):
        return len(self.ids)*2

    def __getitem__(self, index):
        identifier = self.ids[index//2]
        mirror = bool(index % 2)
        length = len(self.source(identifier)[0][0])
        start = random.randrange(length-300) if length > 300 else 0
        anchor, swap = random.choice((1, 2)), bool(random.getrandbits(1))
        x, offsets = self.pair(identifier, start, mirror, anchor, swap)
        text = random.choice(self.adapter.texts(identifier)).strip()
        if mirror:
            text = text.replace('left', 'tmp').replace('right', 'left').replace('tmp', 'right').replace('clockwise', 'tmp').replace('counterclockwise', 'clockwise').replace('tmp', 'counterclockwise')
        n = len(x)
        padded = np.zeros((300, 2, 268), np.float32)
        padded[:n] = x
        return text, padded, n, offsets


def self_check():
    torch.set_num_threads(2)
    data = RootRotationData()
    worst = 0.
    for identifier in data.ids[:8]:
        for mirror in (False, True):
            for anchor in (1, 2):
                for swap in (False, True):
                    x, offsets = data.pair(identifier, 3, mirror, anchor, swap, max_frames=32)
                    value = torch.from_numpy(x)[None].requires_grad_()
                    pred = fk_positions(value, torch.from_numpy(offsets)[None])
                    target = value[..., :66].reshape_as(pred)
                    error = (pred-target).norm(dim=-1).max().item()
                    worst = max(worst, error)
                    assert error < 2e-5, (identifier, mirror, anchor, swap, error)
                    # Isolate FK gradients: no direct-position path to local rotations.
                    (pred[..., 10, :].square().mean()).backward()
                    assert torch.isfinite(value.grad).all()
                    assert value.grad[..., 132:258].abs().sum() > 0
                    assert value.grad[..., 262:].abs().sum() > 0
    print({'status': 'passed', 'variants': 64, 'max_fk_error_m': worst}, flush=True)


if __name__ == '__main__':
    self_check()
