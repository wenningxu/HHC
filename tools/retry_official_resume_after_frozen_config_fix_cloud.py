"""Explicit attempt2 after verified pre-fit frozen-config failure, never a loop."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recover_official_intergen_after_container_restart_cloud import proc, read, sha, write_new

ROOT = Path('/workspace/HHC')
RUN = ROOT/'output/intergen_official_baseline_20261017'
REC = RUN/'recovery_20261020'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    launch = read(RUN/'LAUNCH.json')
    assert launch['pid'] == 997 and not proc(997)
    assert sha(RUN/'LAUNCH.json') == 'b9de0eefc8acfa7679e57ec3da882e76d45f478745969c7b7dc9617361f65630'
    tail = Path(launch['log_path']).read_text()
    assert 'Attempted to set LAST_ITER to 255200, but CfgNode is immutable' in tail
    assert not list((REC/'training_events').glob('*.json'))
    proof = read(REC/'cpu_probe_v2/result.json')
    assert proof['status'] == 'passed' and not proof['cpu_cuda_initialized']
    wrapper = ROOT/'tools/resume_official_intergen_full_state_cloud.py'
    assert sha(wrapper) == '0093c5346d4cedaa76c77d4175b2ad2a7cc21a447a31faf594b2a2fa3072ba0b'
    hhc = read(ROOT/'migration/hhc-after-official-intergen-resume.json')
    for path, expected in hhc['pins'].items():
        if path != str(RUN/'LAUNCH.json'):
            assert sha(path) == expected, path
    catalogue = read(hhc['official_source_catalogue'])
    for relative, expected in catalogue['tracked_files'].items():
        expected = launch.get('source_changes', {}).get(relative, {}).get('after', expected)
        assert sha(Path(launch['cwd'])/relative) == expected, relative
    assert read(REC/'input_restore_acceptance.json')['status'] == 'restored_pinned_inputs_no_gpu_launch'
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not apps.strip()
    archive = REC/'LAUNCH.recovery_attempt1.json'
    with archive.open('xb') as handle:
        handle.write((RUN/'LAUNCH.json').read_bytes())
    write_new(REC/'attempt1_terminal_acceptance.json', {
        'status': 'terminal_pre_fit_no_update', 'pid': 997, 'missing_proc_verified': True,
        'log_sha256': sha(launch['log_path']), 'reason': 'frozen YACS config counter assignment',
        'no_restore_events': True, 'time_unix': time.time()})
    events = REC/'training_events_attempt2'
    events.mkdir()
    command = ['/workspace/intergen-official-env-py38/bin/python', '-u', str(wrapper),
               '--runtime-wrapper', str(ROOT/'tools/run_official_intergen_checkpointed_cloud.py'),
               '--checkpoint-receipt', str(REC/'checkpoint_resume_acceptance.json'),
               '--event-dir', str(events)]
    write_new(REC/'training.attempt2.launch_intent.json', {
        'command': command, 'wrapper_sha256': sha(wrapper), 'time_unix': time.time(),
        'prior_attempt_terminal_acceptance': str(REC/'attempt1_terminal_acceptance.json')})
    descriptor = os.open(REC, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    log_path = RUN/'train.recovery_attempt2_20261020.log'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MPLBACKEND': 'Agg'}
    with log_path.open('x') as log:
        child = subprocess.Popen(command, cwd=launch['cwd'], env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = proc(child.pid)
    assert identity and identity['state'] not in ('Z', 'X')
    launch.update(pid=child.pid, birth=identity['birth'], time=time.time(),
                  command=command, log_path=str(log_path), previous_launch=str(archive))
    launch['runtime_changes']['full_state_resume'].update(wrapper_sha256=sha(wrapper),
        cpu_probe_report=str(REC/'cpu_probe_v2/result.json'), event_dir=str(events),
        logging_counter_restore='after_original_on_train_start; frozen config unchanged')
    temporary = RUN/'LAUNCH.recovery_attempt2.tmp'
    write_new(temporary, launch)
    os.replace(temporary, RUN/'LAUNCH.json')
    result = {'status': 'attempt2_dispatched_pending_update', 'trainer': identity,
              'launch_sha256': sha(RUN/'LAUNCH.json'), 'log_path': str(log_path),
              'wrapper_sha256': sha(wrapper), 'time_unix': time.time()}
    write_new(REC/'dispatch_attempt2_receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
