"""One-shot supervised full-state resume; no automatic retries or model changes."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import time

from restore_official_waiters_after_restart_cloud import ROOT, RUN, read, sha, proc, write_new, atomic


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recovery-dir', type=Path, required=True)
    p.add_argument('--expected-launch-sha', required=True)
    p.add_argument('--training-driver', choices=['resume_official_intergen_gpu1_cloud.py',
                   'resume_official_intergen_single_loader_cloud.py'],
                   default='resume_official_intergen_gpu1_cloud.py')
    a = p.parse_args()
    rec = a.recovery_dir
    assert RUN in rec.resolve().parents
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert sha(RUN/'LAUNCH.json') == a.expected_launch_sha
    previous = read(RUN/'LAUNCH.json')
    old = proc(previous['pid'])
    assert not old or old['birth'] != previous['birth'] or old['state'] in ('Z', 'X')
    receipt = rec/'checkpoint_acceptance.json'
    accepted = read(receipt)
    assert accepted['status'] == 'verified_full_resume_candidate'
    command = ['/workspace/intergen-official-env-py38/bin/python', '-u',
               str(ROOT/'tools'/a.training_driver),
               '--runtime-wrapper', str(ROOT/'tools/run_official_intergen_checkpointed_cloud.py'),
               '--checkpoint-receipt', str(receipt), '--event-dir', str(rec/'training_events')]
    write_new(rec/'LAUNCH.before.json', previous)
    write_new(rec/'launch_intent.json', {'command': command, 'time': time.time(),
              'checkpoint_sha256': accepted['sha256'], 'supervisor_pid': os.getpid()})
    env = dict(os.environ, TMPDIR='/workspace/tmp', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    log = rec/'train.log'
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=previous['cwd'], env=env,
                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    identity = proc(child.pid)
    assert identity and identity['state'] not in ('Z', 'X')
    launch = copy.deepcopy(previous)
    launch.update(pid=child.pid, birth=identity['birth'], command=command,
                  log_path=str(log), time=time.time(), previous_launch=str(rec/'LAUNCH.before.json'),
                  status='supervised_full_resume_pending_acceptance', supervisor_pid=os.getpid())
    launch['runtime_changes']['full_state_resume'].update(
        wrapper=str(ROOT/'tools'/a.training_driver), wrapper_sha256=sha(ROOT/'tools'/a.training_driver),
        resume_step=accepted['step'], checkpoint_receipt=str(receipt),
        checkpoint_receipt_sha256=sha(receipt), checkpoint_sha256=accepted['sha256'],
        event_dir=str(rec/'training_events'))
    if a.training_driver == 'resume_official_intergen_single_loader_cloud.py':
        launch['runtime_changes']['data_loader'] = {
            'num_workers': 0, 'reason': 'isolate observed malloc corruption during multiprocessing receive',
            'root_cause_confirmed': False, 'rng_sequence_equivalence_claimed': False}
    atomic(RUN/'LAUNCH.json', launch)
    code = child.wait()
    write_new(rec/'trainer_exit.json', {'pid': child.pid, 'birth': identity['birth'],
              'returncode': code, 'signal': -code if code < 0 else None, 'time': time.time()})
    raise SystemExit(code if code >= 0 else 128-code)


if __name__ == '__main__':
    main()
