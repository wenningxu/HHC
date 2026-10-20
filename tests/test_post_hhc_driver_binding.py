import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'tools/run_fair_interx_method.py'
PATCH = ROOT / 'tools/provenance_snapshots/post_hhc_timotion_selection_binding.patch'
APPLIER = ROOT / 'tools/apply_post_hhc_timotion_selection_binding.sh'
QUEUE = ROOT / 'tools/queue_post_formal_after_v3.sh'
SNAPSHOT = (
    ROOT / 'output/fair_eval/'
    'hhc_formal_raw12288_repaired_mgt_common50_contact/_provenance_sources/'
    '014a720cdb53eb43bd4834391e8b97bc70763d8b31b43dc00a398b66fdaed6ec'
    '__run_fair_interx_method.py'
)
LAUNCH_SHA = '014a720cdb53eb43bd4834391e8b97bc70763d8b31b43dc00a398b66fdaed6ec'
TIMOTION_SHA = '7b83a6b748dfcff07082973778d188925308164c58ced498504483a68e888d7e'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_hhc_driver_snapshot_and_post_patch_are_exact_and_reversible():
    assert sha256(SNAPSHOT) == LAUNCH_SHA
    current = sha256(DRIVER)
    assert current in {LAUNCH_SHA, TIMOTION_SHA}
    command = ['git', 'apply', '--check']
    if current == TIMOTION_SHA:
        command.append('--reverse')
    command.append(str(PATCH.relative_to(ROOT)))
    subprocess.run(command, cwd=ROOT, check=True)


def test_post_hhc_applier_is_guarded_by_completed_source_stability():
    source = APPLIER.read_text(encoding='utf-8')
    queue = QUEUE.read_text(encoding='utf-8')
    for required in (
        LAUNCH_SHA,
        TIMOTION_SHA,
        'implementation_source_stability_verified_at_completion',
        'Formal HHC common-50 driver is still active',
        'git apply --check',
        'git apply --whitespace=error-all',
    ):
        assert required in source
    assert queue.index(
        'tools/apply_post_hhc_timotion_selection_binding.sh'
    ) < queue.index('tools/queue_post_hhc_gpu1_experiments.sh')


if __name__ == '__main__':
    test_formal_hhc_driver_snapshot_and_post_patch_are_exact_and_reversible()
    test_post_hhc_applier_is_guarded_by_completed_source_stability()
