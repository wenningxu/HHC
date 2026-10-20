"""Small CPU-only regression check; never scores or changes real checkpoints."""
import subprocess
import tempfile
from pathlib import Path

from check_vq_reproduction_gate import no_worse


def main():
    assert no_worse(1., 1.) and no_worse(1., .9) and not no_worse(1., 1.1)
    assert no_worse(1., 1.1, True) and not no_worse(1., .9, True)
    for invalid in (float('nan'), float('inf')):
        try:
            no_worse(1., invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('nonfinite metric accepted')
    script = Path(__file__).with_name('score_vq_seed124_milestones_20260911.sh')
    source = script.read_text()
    function = source[source.index('score_point() {'):source.index('\nvalidate_manifest\nsteps=')]
    function = function.replace('/home/wenning/miniconda3/envs/New_MMM/bin/python', 'fake_python')
    with tempfile.TemporaryDirectory(prefix='hhc-readiness-test-') as directory:
        for check_only in (1, 0):
            setup = (f'set -euo pipefail\nrun_dir={directory}\nprefix={directory}/candidate_\n'
                     f'check_only={check_only}\n'
                     'fake_python() { printf "PY_CALL %s\\n" "$1"; }\n')
            result = subprocess.run(['bash'], input=setup + function + '\nscore_point 5000\n',
                                    text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            calls = [line for line in result.stdout.splitlines() if line.startswith('PY_CALL ')]
            assert len(calls) == (1 if check_only else 4), calls
            assert calls[0] == 'PY_CALL -'
    print('PASS: metric directions/nonfinite rejection; ready check-only never scores')


if __name__ == '__main__':
    main()
