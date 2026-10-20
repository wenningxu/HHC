import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


DIAG = module('mm_diag', 'tools/diagnose_mm_repeat_collapse.py')
CARD = module('amass_card_mm', 'tools/build_amass_shape_augmentation_result_card.py')


def test_repeat_collapse_binds_h5_and_implementation(tmp_path):
    label = 'synthetic_common50'
    h5_path = tmp_path / 'replication_00_mm.h5'
    motions = np.zeros((100, 30, 2, 1), dtype=np.float32)
    motions[:, :, 0, 0] = np.arange(30, dtype=np.float32)[None, :]
    with h5py.File(h5_path, 'w') as target:
        target.create_dataset('motions', data=motions)
        target.create_dataset('lengths', data=np.full(100, 2, dtype=np.int64))
        target.attrs['sampler'] = 'sample_intra_cas'
        target.attrs['sampling_steps'] = 50
        target.attrs['temperature'] = 0.5
        target.attrs['cfg_scale'] = 3.5
    output = tmp_path / 'collapse.json'
    previous = sys.argv
    try:
        sys.argv = [
            'diagnose', '--input', f'{label}={h5_path}', '--output', str(output),
            '--pairs-per-prompt', '5',
        ]
        with redirect_stdout(io.StringIO()):
            DIAG.main()
    finally:
        sys.argv = previous
    compact = CARD.validate_repeat_collapse(output, label, h5_path)
    assert compact['exact_pair_rate'] == 0.0
    assert compact['unique_repeats_per_prompt']['minimum'] == 30
    assert list(tmp_path.glob('collapse.json.tmp.*')) == []

    with h5py.File(h5_path, 'a') as target:
        target.attrs['mutated_after_diagnostic'] = True
    try:
        CARD.validate_repeat_collapse(output, label, h5_path)
    except ValueError:
        pass
    else:
        raise AssertionError('stale MM H5 evidence was accepted')
