"""CPU check: shared alignment preserves relative geometry and round trips."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.interx_ground_alignment import (
    align_pair_for_mmm_optimization, align_person_for_mmm_optimization,
    restore_person_to_raw_coordinates, SHARED_GROUND_ALIGNMENT_POLICY,
)


def geometry(p):
    joints = p['trans'][:, None, :] + np.array([[[0., -p['height'], 0.]]], dtype=np.float32)
    return joints, joints


if __name__ == '__main__':
    a = {'trans': np.array([[0., .2, 0.], [.1, .3, .1]], dtype=np.float32), 'height': 1.}
    b = {'trans': np.array([[.4, .1, 0.], [.3, .2, .1]], dtype=np.float32), 'height': 1.4}
    originals = [p['trans'].copy() for p in (a, b)]
    old_a, fa, old_b, fb = align_pair_for_mmm_optimization(a, b, geometry)
    for p, got, floor in ((a, old_a, fa), (b, old_b, fb)):
        expected, ef = align_person_for_mmm_optimization(p, geometry)
        assert floor == ef and np.array_equal(got['trans'], expected['trans'])
    aa, ca, bb, cb = align_pair_for_mmm_optimization(a, b, geometry, SHARED_GROUND_ALIGNMENT_POLICY)
    assert ca == cb == min(fa, fb)
    np.testing.assert_allclose(geometry(aa)[0] - geometry(bb)[0], geometry(a)[0] - geometry(b)[0], atol=2e-7)
    for original, saved, aligned, floor in zip((a, b), originals, (aa, bb), (ca, cb)):
        np.testing.assert_allclose(restore_person_to_raw_coordinates(aligned, floor)['trans'], saved, atol=2e-7)
        assert np.array_equal(original['trans'], saved)
    try:
        align_pair_for_mmm_optimization(a, b, geometry, 'invalid')
        raise AssertionError('invalid policy accepted')
    except ValueError:
        pass
    print('PASS: old parity, relative geometry, shared floor, roundtrip, no input mutation, policy validation')
