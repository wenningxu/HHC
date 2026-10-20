import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_geometry_relation_timing_pilot import geometric_times


def check():
    w = geometric_times([[0., .04], [.02, .01], [.04, .02]])
    assert np.allclose(w, [[1., 0.], [.5, 1.], [0., .5]])
    for invalid in ([[.04]], [[-1.]], [[float('nan')]], []):
        try:
            geometric_times(invalid)
        except AssertionError:
            continue
        raise AssertionError('invalid or unsupported geometry accepted')


if __name__ == '__main__':
    check()
