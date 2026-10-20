"""Recover only the two never-dispatched downstream CPU validation waiters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recover_official_intergen_after_container_restart_cloud import proc, read, sha, write_new

ROOT = Path('/workspace/HHC')
REC = ROOT/'output/intergen_official_baseline_20261017/recovery_20261020'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    write_new(REC/'downstream_recovery_intent.json', {'time_unix': time.time(), 'scope': 'pure waiting only'})
    records = {}
    entries = [('hhc-text-only-validation', 'run_hhc_text_only_validation_cloud.py', 'waiting_for_all_mgt'),
               ('hhc-train-shape-validation-v2', 'run_hhc_train_shape_validation_v2_cloud.py', 'waiting_for_zero_shape_scoring')]
    for name, source, waiting in entries:
        config_path = ROOT/'migration'/(name+'.json')
        config = read(config_path)
        control = Path(config['control'])
        assert ROOT/'output' in control.resolve().parents
        status_path = control/'STATUS.json'
        previous = read(status_path)
        assert previous['status'] == waiting and previous['config_sha256'] == sha(config_path)
        assert not previous.get('phase') and not previous.get('child_pid') and not previous.get('phase_exit_codes')
        assert not proc(previous['pid'])
        assert not list(control.glob('*intent*'))
        assert not list(control.glob('*_generate.log')) and not list(control.glob('*_score.log'))
        for relative, expected in config['sources'].items():
            assert sha(ROOT/relative) == expected, relative
        if 'shape_prior' in config:
            assert sha(config['shape_prior']) == config['shape_prior_spec_sha256']
            prior = read(config['shape_prior'])
            assert sha(prior['array']) == prior['array_sha256'] == config['shape_array_sha256']
        else:
            assert any(not Path(a['run'], 'training_manifest.json').exists()
                       or read(Path(a['run'])/'training_manifest.json')['status'] != 'pilot_complete_validation_only'
                       for a in config['arms'])
        archive = control/'STATUS.before_restart_20261020.json'
        assert not archive.exists()
        status_path.rename(archive)
        command = [sys.executable, '-u', str(ROOT/'tools'/source), '--config', str(config_path)]
        with (control/'recovery_20261020.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(50):
            if status_path.exists():
                state = read(status_path)
                if state.get('pid') == child.pid:
                    break
            time.sleep(.1)
        else:
            raise RuntimeError('Recovered waiter did not publish its own status')
        identity = proc(child.pid)
        assert identity and identity['state'] not in ('Z', 'X') and state['status'] == waiting
        records[name] = dict(identity, status=waiting, config_sha256=sha(config_path))
        write_new(control/'recovery_launch_receipt_20261020.json', records[name])
    result = {'status': 'downstream_pure_waiters_recovered', 'identities': records,
              'no_validation_phase_replayed': True, 'no_gpu_child_launched': True, 'time_unix': time.time()}
    write_new(REC/'downstream_recovery_receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
