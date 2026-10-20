import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dataset.interhuman_generated_h5 import (
    InterHumanGeneratedH5Dataset,
    InterHumanMMGeneratedH5Dataset,
    mm_repeat_seed,
)
from tools.export_hhc_interhuman import initialize_or_validate
from tools.aggregate_fair_interhuman_metrics import _identity


def _args(mode):
    return SimpleNamespace(
        mode=mode, sampling_steps=50, temperature=0.5, cfg_scale=3.5,
        batch_size=8,
    )


def _provenance():
    return {
        'checkpoint': '/checkpoint.pth', 'checkpoint_sha256': 'mgt-hash',
        'training_manifest': '/manifest.json',
        'training_manifest_sha256': 'manifest-hash',
        'training_identity_sha256': 'identity',
        'vq': {'checkpoint': '/vq.pth', 'checkpoint_sha256': 'vq-hash'},
        'model_contract': {'architecture': 'test'},
    }


def _request(count, repeats=None):
    value = {
        'request_sha256': 'request-content',
        'requests': [
            {
                'identifier': str(index), 'caption': f'text {index}',
                'processed_length': 100, 'sample_seed': index + 100,
            }
            for index in range(count)
        ],
    }
    if repeats is not None:
        value['mm_repeats'] = repeats
    return value


def test_main_export_schema_is_formal_request_bound_and_resumable():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        request = _request(1056)
        request_path = root / 'requests.json'
        request_path.write_text(json.dumps(request))
        output = root / 'main.h5'
        handle = initialize_or_validate(
            output, _args('main'), request_path, request, _provenance()
        )
        assert handle['motion1'].shape == (1056, 300, 262)
        assert handle['completed'].shape == (1056,)
        assert not np.asarray(handle['completed'][:]).any()
        assert handle.attrs['model_config_sha256']
        assert handle.attrs['generation_exporter_sha256']
        assert handle.attrs['interhuman_generation_bridge_sha256']
        assert not bool(handle.attrs['mgt_shape_condition'])
        assert not bool(handle.attrs['contact_conditioning'])
        handle.close()
        resumed = initialize_or_validate(
            output, _args('main'), request_path, request, _provenance()
        )
        resumed.close()
        validated = InterHumanGeneratedH5Dataset(output, request_path, formal=True)
        assert len(validated) == 1056
        identity = _identity(validated.metadata, require_hhc_common50=True)
        assert identity['sampler'] == 'sample_intra_cas'
        validated.close()


def test_mm_export_freezes_exact_uint64_repeat_seed_matrix():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        request = _request(100, repeats=30)
        request_path = root / 'requests.json'
        request_path.write_text(json.dumps(request))
        output = root / 'mm.h5'
        handle = initialize_or_validate(
            output, _args('mm'), request_path, request, _provenance()
        )
        assert handle['motion1'].shape == (100, 30, 300, 262)
        assert handle['repeat_seeds'].dtype == np.uint64
        assert int(handle['repeat_seeds'][7, 29]) == mm_repeat_seed(107, 29)
        handle.close()
        validated = InterHumanMMGeneratedH5Dataset(output, request_path, formal=True)
        assert len(validated) == 100 and validated.repeat_count == 30
        validated.close()


def test_interhuman_finalizer_locks_twenty_common50_main_and_mm_replications():
    source = Path('tools/finalize_hhc_interhuman_common50.sh').read_text()
    assert 'seq -w 0 19' in source
    assert source.count('--sampling-steps 50 --temperature 0.5 --cfg-scale 3.5') == 2
    assert source.count('--require-hhc-common50') == 2
    assert 'tools/score_fair_interhuman_main.py' in source
    assert 'tools/score_fair_interhuman_mm.py' in source
    assert 'tools/aggregate_fair_interhuman_metrics.py' in source
    assert 'tools/audit_wandb_online_metric_history.py' in source
    assert '--metric-profile interhuman_mgt' in source
    assert '--expected-final-step 300000' in source
