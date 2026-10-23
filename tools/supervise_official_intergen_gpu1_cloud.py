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
    p.add_argument('--native-debug', action='store_true')
    p.add_argument('--asan', action='store_true',
                   help='Preload ASAN only into the GDB inferior; requires --native-debug')
    a = p.parse_args()
    assert not a.asan or a.native_debug
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
    trainer_command = command[:]
    if a.native_debug:
        identity_file = rec/'inferior_identity.json'
        assert not identity_file.exists()
        trainer_command = command[:2] + [str(ROOT/'tools/gdb_official_intergen_inferior_cloud.py'),
            '--identity-file', str(identity_file), '--driver', command[2]] + command[3:]
        debug_environment = []
        if a.asan:
            assert os.environ.get('PYTHONMALLOC') == 'malloc'
            assert not os.environ.get('LD_PRELOAD'), 'Do not instrument the supervisor or debugger'
            for library in ('libasan.so.8', 'libstdc++.so.6'):
                assert (Path('/lib/x86_64-linux-gnu') / library).is_file()
            debug_environment = [
                '-ex', 'set environment LD_PRELOAD /lib/x86_64-linux-gnu/libasan.so.8:/lib/x86_64-linux-gnu/libstdc++.so.6',
                '-ex', 'set environment ASAN_OPTIONS detect_leaks=0:halt_on_error=1:abort_on_error=1:quarantine_size_mb=128:protect_shadow_gap=0']
        command = ['/usr/bin/gdb', '--batch', '--return-child-result',
                   '-ex', 'set pagination off'] + debug_environment + ['-ex', 'run',
                   '-ex', 'thread apply all bt 32', '-ex', 'info sharedlibrary',
                   '--args'] + trainer_command
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
    if a.native_debug:
        deadline = time.monotonic()+60
        while not identity_file.exists() and child.poll() is None and time.monotonic()<deadline:
            time.sleep(.1)
        if not identity_file.exists():
            # Preserve a possibly initializing debugger; never silently retry it.
            write_new(rec/'debugger_startup_attention.json', {'pid': child.pid,
                'returncode': child.poll(), 'time': time.time()})
            raise RuntimeError('Debugger has not registered its inferior; inspect the existing process')
        registered = read(identity_file)
        identity = proc(registered['pid'])
        assert identity and identity['birth'] == registered['birth']
    assert identity and identity['state'] not in ('Z', 'X')
    launch = copy.deepcopy(previous)
    launch.update(pid=identity['pid'], birth=identity['birth'], command=trainer_command,
                  log_path=str(log), time=time.time(), previous_launch=str(rec/'LAUNCH.before.json'),
                  status='supervised_full_resume_pending_acceptance', supervisor_pid=os.getpid())
    if a.native_debug:
        launch['native_debugger'] = {'pid': child.pid, 'command': command,
            'inferior_wrapper_sha256': sha(ROOT/'tools/gdb_official_intergen_inferior_cloud.py')}
    if a.asan:
        launch['runtime_changes']['asan'] = {
            'enabled': True, 'inferior_only': True,
            'scope': 'Preload interceptors; uninstrumented native direct writes not fully covered',
            'root_cause_fixed': False, 'settings': debug_environment}
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
    write_new(rec/'trainer_exit.json', {'pid': identity['pid'], 'birth': identity['birth'],
              'returncode': code, 'signal': -code if code < 0 and not a.native_debug else None,
              'returncode_source': 'gdb --return-child-result' if a.native_debug else 'trainer subprocess',
              'time': time.time()})
    raise SystemExit(code if code >= 0 else 128-code)


if __name__ == '__main__':
    main()
