import tempfile
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_paper_contact_optimization import output_payload
from tools.materialize_interx_shape_augmentation import output_raw
from dataset.dataset_interx_tokenize import Text2MotionDatasetV2HHI
from utils.interx_physical_ground import encode_physical_pair, decode_physical_pair
from utils.motion_process import MotionNormalizerTorchInterX


def run():
    n = 3
    people = []
    for i in range(2):
        people.append({'root_orient': np.zeros((n, 3), np.float32), 'pose_body': np.zeros((n, 21, 3), np.float32),
                       'pose_lhand': np.zeros((n, 15, 3), np.float32), 'pose_rhand': np.zeros((n, 15, 3), np.float32),
                       'trans': (np.arange(n * 3, dtype=np.float32).reshape(n, 3) + i * 2),
                       'betas': np.zeros(10, np.float32), 'gender': 'male'})
    enc, origin = encode_physical_pair(people)
    assert np.allclose(enc[1]['trans'] - enc[0]['trans'], people[1]['trans'] - people[0]['trans'])
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, p in enumerate(enc):
            path = f'{d}/p{i}.npz'; np.savez(path, **output_payload(p, p['gender'])); paths.append(path)
        raw, _, _ = output_raw(*paths)
    x = Text2MotionDatasetV2HHI.__new__(Text2MotionDatasetV2HHI); x.num_person = 2
    motion = x.to_rot_6d(raw); assert motion.shape[0] == n
    normalizer = MotionNormalizerTorchInterX('raw')
    y = normalizer.forward(motion).requires_grad_(); y.sum().backward()
    restored = normalizer.backward(y.detach()).reshape(n, 56, 12)
    assert torch.isfinite(y).all() and torch.isfinite(restored).all()
    recovered = [dict(enc[i], trans=restored[:, 55, 6 * i:6 * i + 3].numpy()) for i in range(2)]
    dec = decode_physical_pair(recovered, origin)
    assert np.allclose(dec[0]['trans'], people[0]['trans']) and np.allclose(dec[1]['trans'], people[1]['trans'])
    assert np.allclose(recovered[1]['trans'] - recovered[0]['trans'], people[1]['trans'] - people[0]['trans'])


if __name__ == '__main__':
    run(); print('PASS')
