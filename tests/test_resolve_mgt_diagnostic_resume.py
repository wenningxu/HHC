import json
from pathlib import Path

from tools.resolve_mgt_diagnostic_resume import resolve


def make_run(root: Path, experiment: str, timestamp='2026-01-01-00-00-00'):
    run = root / f'{timestamp}_{experiment}'
    run.mkdir()
    (run / 'training_manifest.json').write_text(json.dumps({'status': 'running'}))
    return run


def assert_runtime_error(function):
    try:
        function()
    except RuntimeError:
        return
    raise AssertionError('expected RuntimeError')


def test_new_when_no_matching_directory(tmp_path):
    assert resolve(tmp_path, 'experiment', 50000)['action'] == 'new'


def test_resume_uses_highest_rolling_checkpoint_below_stop(tmp_path):
    run = make_run(tmp_path, 'experiment')
    for step in (10000, 30000, 40000):
        (run / f'latest-{step:09d}.ckpt').touch()
    result = resolve(tmp_path, 'experiment', 50000)
    assert result['action'] == 'resume'
    assert result['resume_step'] == 40000
    assert result['resume_checkpoint'].endswith('latest-000040000.ckpt')


def test_complete_requires_both_rolling_and_validation_checkpoint(tmp_path):
    run = make_run(tmp_path, 'experiment')
    (run / 'latest-000050000.ckpt').touch()
    assert_runtime_error(lambda: resolve(tmp_path, 'experiment', 50000))
    (run / 'model-000050000-0.1234.ckpt').touch()
    result = resolve(tmp_path, 'experiment', 50000)
    assert result['action'] == 'complete'
    assert result['validation_checkpoint'].endswith(
        'model-000050000-0.1234.ckpt'
    )


def test_ambiguous_duplicate_runs_fail_closed(tmp_path):
    make_run(tmp_path, 'experiment', '2026-01-01-00-00-00')
    make_run(tmp_path, 'experiment', '2026-01-02-00-00-00')
    assert_runtime_error(lambda: resolve(tmp_path, 'experiment', 50000))
