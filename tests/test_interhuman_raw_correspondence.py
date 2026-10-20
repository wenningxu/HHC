"""Local regression checks against the released raw/processed sample 7778."""
import pickle
from pathlib import Path

import numpy as np
import torch

from tools.reconstruct_interhuman_raw_body import load_model, reconstruct_person
from tools.download_interhuman_raw import valid_name, verified


def test_download_names():
    assert valid_name('7778.pkl')
    assert not valid_name('../7778.pkl')
    assert not valid_name('7778.pkl/other')


def test_raw_body_release_correspondence():
    root = Path('dataset/interhuman_official')
    raw = pickle.loads((root / 'motions/7778.pkl').read_bytes())
    model = load_model()
    for who in ('person1', 'person2'):
        p = raw[who]
        a, idx = reconstruct_person(p, raw['mocap_framerate'], model)
        assert idx[:6].tolist() == [0, 1, 3, 5, 7, 9]
        b = np.load(root / 'motions_processed' / who / '7778.npy')
        assert len(a) == len(b) == 79
        assert np.abs(a[:, :66]-b[:, :66]).max() < 1e-5
        assert np.abs(a[:, 66:]-b[:, 186:312]).max() < 1e-6
        # Optimized joint-only FK must agree with standard full SMPL forward.
        with torch.no_grad():
            full = model(global_orient=torch.from_numpy(p['root_orient'][idx[:3]]),
                         body_pose=torch.from_numpy(np.pad(p['pose_body'][idx[:3]], ((0, 0), (0, 6)))),
                         transl=torch.from_numpy(p['trans'][idx[:3]]),
                         betas=torch.from_numpy(p['betas'][None]).expand(3, -1))
        assert np.max(np.abs(full.joints[:, :22].numpy().reshape(3, 66)-a[:3, :66])) < 1e-6
