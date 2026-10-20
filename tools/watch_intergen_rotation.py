"""Hourly, single-instance supervision of the authorized InterGen fine-tune."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/home/wenning/HHC_interhuman_runs/intergen_rotation_fk_seed20261008')
LOG = ROOT/'output/intergen_rotation_fk_20261008'
PYTHON = '/home/wenning/miniconda3/envs/New_MMM/bin/python'
TOTAL_STEPS = 10000


def completed(code, status):
    return code == 0 and status.get('status') == 'complete' and status.get('step') == TOTAL_STEPS


def status():
    try:
        return json.loads((RUN/'STATUS.json').read_text())
    except (OSError, ValueError):
        return {}


def record(event, **values):
    row = {'time': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'event': event, **values}
    with (LOG/'watchdog.jsonl').open('a') as handle:
        handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def main():
    global RUN, LOG, TOTAL_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', action='store_true')
    args = parser.parse_args()
    if args.scratch:
        RUN = ROOT/'output/intergen_rotation_scratch_20261010'
        LOG = ROOT/'output/intergen_rotation_scratch_monitor_20261010'
        TOTAL_STEPS = 704000
    RUN.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    lock = (LOG/'watchdog.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (LOG/'watchdog.pid').write_text(str(os.getpid())+'\n')
    child = None
    while True:
        current = status()
        if child is not None:
            code = child.poll()
            if code is not None:
                record('trainer_exit', pid=child.pid, exit_code=code, status=current)
                if completed(code, current):
                    record('completed', status=current)
                    return
                child = None
        elif current.get('status') == 'complete' and current.get('step') == TOTAL_STEPS:
            record('already_completed', status=current)
            return
        if child is None:
            # Do not start a second trainer if another supervisor owns the run.
            with (RUN/'.training.lock').open('a') as training_lock:
                try:
                    fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    record('existing_trainer_not_touched', status=current)
                    time.sleep(3600)
                    continue
            attempt_log = LOG/f'resume_{time.time_ns()}.log'
            env = dict(os.environ, PYTHONFAULTHANDLER='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            command = [PYTHON, '-u', str(ROOT/'tools/train_intergen_rotation.py'), '--output', str(RUN)]
            if args.scratch:
                command.append('--scratch')
                if (RUN/'recovery.pt').exists():
                    command.append('--resume')
            else:
                command.append('--resume')
            with attempt_log.open('x') as handle:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            (LOG/'trainer.pid').write_text(str(child.pid)+'\n')
            record('trainer_started', pid=child.pid, log=str(attempt_log), interval_seconds=3600)
        else:
            evaluations = sorted(RUN.glob('validation_*/validation_metrics.json'))
            latest = None
            if evaluations:
                try:
                    latest = {'path': str(evaluations[-1]), 'metrics': json.loads(evaluations[-1].read_text())['arms']['raw']}
                except (OSError, ValueError, KeyError):
                    pass
            record('hourly_check', pid=child.pid, status=current, latest_validation=latest)
        time.sleep(3600)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        assert completed(0, {'status': 'complete', 'step': 10000})
        assert not completed(132, {'status': 'training', 'step': 2300})
        assert not completed(0, {'status': 'training', 'step': 2300})
        assert not completed(0, {'status': 'complete', 'step': 2000})
        print('completion checks passed')
    else:
        main()
