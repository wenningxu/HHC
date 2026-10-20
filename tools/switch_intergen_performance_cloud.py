"""Cloud CPU coordinator: verified recovery -> performance-resumed InterGen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from intergen_speed_probe_cloud import proc, same_live, write
from pause_hhc_for_intergen_cloud import sources, terminal

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/intergen_rotation_scratch_runpod_20261015'
FAST = ROOT / 'tools/train_intergen_rotation_fast_cloud.py'


def trainers():
    found = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
            if (str(RUN).encode() in args and
                    any(Path(arg.decode(errors='replace')).name in
                        ('train_intergen_rotation.py', 'train_intergen_rotation_fast_cloud.py')
                        for arg in args if arg)):
                identity = proc(int(p.parent.name))
                if identity and identity['state'] not in ('Z', 'X'):
                    found.append(identity)
        except FileNotFoundError:
            pass
    return found


def start_fast(folder, name):
    with (folder / name).open('a') as log:
        child = subprocess.Popen(['bash', str(ROOT / 'tools/runpod_fast_bootstrap.sh')],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0',
                                      'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                                      'OPENBLAS_NUM_THREADS': '2'})
    return proc(child.pid)


def guard(folder):
    while not (folder / 'coordinator_done.json').exists():
        identity = json.loads((folder / 'guard.json').read_bytes())
        if not same_live(identity['coordinator']) or time.time() > identity['deadline_unix']:
            live = trainers()
            if live:
                for p in live:
                    if p['state'] in ('T', 't'):
                        os.kill(p['pid'], signal.SIGCONT)
                action = {'status': 'existing_trainer_preserved', 'trainers': live}
            else:
                action = {'status': 'fast_resume_rescue_started',
                          'trainer': start_fast(folder, 'rescue_training.log')}
            write(folder / 'guard_result.json', action)
            return
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--minimum-step', type=int, default=84000)
    parser.add_argument('--guard', action='store_true')
    args = parser.parse_args()
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    folder = args.output.resolve()
    assert folder.is_relative_to(ROOT / 'output')
    if args.guard:
        guard(folder)
        return
    manifest = json.loads(args.manifest.read_bytes())

    def unchanged():
        for item in manifest:
            data = (ROOT / item['path']).read_bytes()
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
        policy = json.loads((ROOT / 'migration/hhc-compute-priority.json').read_bytes())
        sources(policy)

    unchanged()
    original = proc(2353)
    assert original and 'tools/train_intergen_rotation.py ' in original['command']
    assert len(trainers()) == 1 and original['state'] not in ('T', 't', 'Z', 'X')
    folder.mkdir(exist_ok=False)
    write(folder / 'previous_config.json', json.loads((RUN / 'config.json').read_bytes()))
    # Arm recovery before the first SIGSTOP, including the checkpoint wait.
    write(folder / 'guard.json', {'coordinator': proc(os.getpid()), 'deadline_unix': time.time()+3600,
                                 'original': original})
    with (folder / 'guard.log').open('w') as log:
        watcher = subprocess.Popen([sys.executable, str(Path(__file__)), '--guard', '--output', str(folder)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    write(folder / 'guard_identity.json', proc(watcher.pid))
    write(folder / 'STATUS.json', {'status': 'waiting_for_complete_recovery', 'minimum_step': args.minimum_step})
    deadline = time.monotonic()+2100
    paused = False
    while True:
        assert same_live(original), 'Original trainer exited before switch'
        assert time.monotonic() < deadline, 'No eligible checkpoint within35minutes'
        try:
            receipt = json.loads((RUN / 'RECOVERY.json').read_bytes())
            status = json.loads((RUN / 'STATUS.json').read_bytes())
        except (FileNotFoundError, ValueError):
            time.sleep(.25)
            continue
        if receipt['step'] >= args.minimum_step and status['step'] == receipt['step']:
            guard_state = json.loads((folder / 'guard.json').read_bytes())
            guard_state['deadline_unix'] = time.time()+900
            write(folder / 'guard.json', guard_state)
            os.kill(2353, signal.SIGSTOP)
            paused = True
            time.sleep(.1)
            new_receipt = json.loads((RUN / 'RECOVERY.json').read_bytes())
            new_status = json.loads((RUN / 'STATUS.json').read_bytes())
            if new_receipt == receipt and new_status['step'] == receipt['step']:
                break
            os.kill(2353, signal.SIGCONT)
            paused = False
            guard_state['deadline_unix'] = time.time()+3600
            write(folder / 'guard.json', guard_state)
        time.sleep(.25)
    write(folder / 'selected_recovery.json', receipt)
    committed = False
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('Coordinator terminated')))
        unchanged()
        import torch
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()
        recovery = RUN / 'recovery.pt'
        stat = recovery.stat()
        with recovery.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert stat.st_size == receipt['bytes'] and digest == receipt['sha256']
        saved = torch.load(recovery, map_location='cpu', mmap=True, weights_only=False)
        assert saved['step'] == saved['checkpoint']['step'] == receipt['step']
        c = saved['checkpoint']['config']
        assert c['effective_batch'] == 32 and c['steps'] == 704000 and c['scratch']
        assert c['micro_batch'] == 4 and c['accumulation'] == 8
        assert {'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng', 'optimizer'} <= saved.keys()
        assert all(torch.isfinite(t).all() for t in saved['checkpoint']['trainable_state'].values())
        assert all(torch.isfinite(t).all() for v in saved['optimizer']['state'].values()
                   for t in v.values() if isinstance(t, torch.Tensor))
        steps = [int(s['step']) for s in saved['optimizer']['state'].values()]
        write(folder / 'recovery_acceptance.json', {'step': receipt['step'], 'bytes': stat.st_size,
                                                    'sha256': digest, 'optimizer_states': len(steps),
                                                    'optimizer_step_min': min(steps), 'optimizer_step_max': max(steps),
                                                    'cpu_only': True, 'source_micro_batch': 4, 'target_micro_batch': 16,
                                                    'effective_batch': 32, 'rng_present': True,
                                                    'loader_cursor_saved': False})
        del saved
        assert not torch.cuda.is_initialized()
        unchanged()
        write(folder / 'STATUS.json', {'status': 'recovery_verified_switching', 'step': receipt['step']})
        assert same_live(original) and proc(2353)['state'] in ('T', 't')
        os.kill(2353, signal.SIGTERM)
        os.kill(2353, signal.SIGCONT)
        paused = False
        limit = time.monotonic()+30
        while same_live(original):
            assert time.monotonic() < limit, 'Original did not terminate'
            time.sleep(.2)
        new = start_fast(folder, 'fast_training.log')
        write(folder / 'fast_launch.json', new)
        committed = True
        limit = time.monotonic()+600
        while True:
            assert time.monotonic() < limit, 'Fast startup/100update acceptance timed out'
            live = trainers()
            assert live, 'Fast trainer exited; guard will restart full-state training'
            try:
                status = json.loads((RUN / 'STATUS.json').read_bytes())
                c = json.loads((RUN / 'config.json').read_bytes())
                if (status.get('status') == 'training' and status.get('step', 0) >= receipt['step']+100
                        and c.get('trainer_source_sha256') == hashlib.sha256(FAST.read_bytes()).hexdigest()
                        and c['micro_batch'] == 16 and c['accumulation'] == 2):
                    break
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(2)
        write(folder / 'STATUS.json', {'status': 'fast_updates_verified_installing_hhc_waiter',
                                       'step': status['step'], 'trainer': live})
        old_control = ROOT / 'output/hhc_after_intergen_resume_control_20261017/STATUS.json'
        old = json.loads(old_control.read_bytes())
        assert old['pid'] == 17952 and old['status'] == 'waiting_for_intergen_complete'
        assert not old.get('phase') and not old.get('phase_exit_codes')
        old_identity = proc(17952)
        if old_identity and same_live(old_identity):
            assert 'resume_hhc_after_intergen_cloud.py' in old_identity['command']
            os.kill(17952, signal.SIGTERM)
            limit = time.monotonic()+10
            while same_live(old_identity):
                assert time.monotonic() < limit
                time.sleep(.1)
        old.update(status='superseded_by_performance_waiter', superseding_control='/workspace/HHC/output/hhc_after_intergen_fast_resume_control_20261017')
        write(old_control, old)
        waiter = ROOT / 'tools/resume_hhc_after_intergen_fast_cloud.py'
        config = ROOT / 'migration/hhc-after-fast-intergen-resume.json'
        for flags in (['--check-only'], ['--detach']):
            result = subprocess.run([sys.executable, str(waiter), '--config', str(config), *flags],
                                    capture_output=True, text=True, check=True, timeout=30)
            write(folder / ('waiter_preflight.json' if flags[0] == '--check-only' else 'waiter_launch.json'),
                  {'stdout': result.stdout, 'exit_code': result.returncode})
        limit = time.monotonic()+30
        while True:
            path = ROOT / 'output/hhc_after_intergen_fast_resume_control_20261017/STATUS.json'
            if path.exists():
                waiting = json.loads(path.read_bytes())
                identity = proc(waiting['pid'])
                if (waiting['status'] == 'waiting_for_intergen_complete' and identity
                        and identity['state'] not in ('Z', 'X') and waiting['saved_hhc_step'] == 4000):
                    break
            assert time.monotonic() < limit, 'New HHC waiter did not become active'
            time.sleep(.2)
        write(folder / 'STATUS.json', {'status': 'performance_continuation_active', 'step': status['step'],
                                       'trainer': trainers(), 'resume_step': receipt['step'],
                                       'hhc_waiter_pid': waiting['pid'],
                                       'effective_batch': 32, 'micro_batch': 16, 'accumulation': 2})
        write(folder / 'coordinator_done.json', {'status': 'success', 'unix': time.time()})
        print((folder / 'STATUS.json').read_text(), flush=True)
    finally:
        if paused and same_live(original):
            os.kill(2353, signal.SIGCONT)
        if not committed and not trainers():
            start_fast(folder, 'fallback_training.log')
        # On an exception after launch the guard remains to preserve/restart
        # training; root can repair the CPU waiting queue without losing it.
        if (folder / 'coordinator_done.json').exists():
            watcher.wait(timeout=10)


if __name__ == '__main__':
    main()
