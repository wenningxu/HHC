"""Save-boundary pause of the existing A run for explicit InterGen priority."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def process(pid, required):
    root = Path('/proc') / str(pid)
    command = (root / 'cmdline').read_bytes()
    assert all(item.encode() in command for item in required), (pid, 'identity mismatch')
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z', (pid, 'zombie')
    return fields[19]


def terminal(pid):
    path = Path('/proc') / str(pid) / 'stat'
    return not path.exists() or path.read_text().rsplit(')', 1)[1].split()[0] == 'Z'


def sources(config):
    expected = {}
    for name in config['bound_configs']:
        for path, value in read(name)['sources'].items():
            assert path not in expected or expected[path] == value
            expected[path] = value
    assert len(expected) == 29
    assert all(sha(ROOT / path) == value for path, value in expected.items())
    return expected


def preflight(config):
    assert str(ROOT) == '/workspace/HHC'
    assert config['policy'] == 'intergen_exclusive_until_complete704000'
    run = Path(config['hhc_run'])
    manifest = read(run / 'training_manifest.json')
    queue = read(config['abc_status'])
    assert manifest['contract_sha256'] == config['hhc_contract_sha256']
    assert manifest['status'] == 'training' and manifest['pid'] == config['hhc_pid']
    assert queue['status'] == 'running' and queue['phase'] == 'A_historical'
    assert queue['pid'] == config['abc_pid'] and queue['child_pid'] == config['hhc_pid']
    assert queue['phase_exit_codes'] == {'freeze_final': 0}
    identity = process(config['hhc_pid'], ['train_hhc_smpl_cfg_ablation.py', str(run)])
    process(config['abc_pid'], ['run_hhc_abc_after_vq_cloud.py'])
    process(config['intergen_pid'], ['train_intergen_rotation.py', config['intergen_run']])
    assert sha(config['vq_spec']) == config['vq_spec_sha256']
    sources(config)
    return identity, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    identity, manifest = preflight(config)
    if not args.execute:
        print(json.dumps({'status': 'preflight_passed_no_signals', 'manifest_step': manifest['step'],
                          'minimum_checkpoint_step': config['minimum_checkpoint_step']}))
        return
    control = Path(config['pause_control'])
    control.mkdir(exist_ok=True)
    if args.detach:
        with (control / 'launch_receipt.json').open('x') as receipt, (control / 'launch.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '--config', str(args.config.resolve()), '--execute'],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
            json.dump({'pid': child.pid, 'config_sha256': sha(args.config), 'started': time.time()}, receipt)
        print(json.dumps({'pid': child.pid, 'status': 'waiting_for_safe_checkpoint'}))
        return
    with (control / 'pause.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (control / 'STATUS.json').exists(), 'Never replay a pause operation'
        state = {'status': 'waiting_for_safe_checkpoint', 'pid': os.getpid(), 'started': time.time(),
                 'config_sha256': sha(args.config), 'minimum_checkpoint_step': config['minimum_checkpoint_step'],
                 'hhc_pid': config['hhc_pid'], 'intergen_preserved': True, 'local_checkpoint_downloads': False}
        write(control / 'STATUS.json', state)
        stopped = False
        try:
            run = Path(config['hhc_run'])
            while True:
                assert process(config['hhc_pid'], ['train_hhc_smpl_cfg_ablation.py', str(run)]) == identity
                process(config['intergen_pid'], ['train_intergen_rotation.py', config['intergen_run']])
                manifest = read(run / 'training_manifest.json')
                assert manifest['status'] == 'training'
                assert manifest['contract_sha256'] == config['hhc_contract_sha256']
                if manifest['step'] >= config['minimum_checkpoint_step']:
                    break
                time.sleep(5)
            sources(config)
            os.kill(config['hhc_pid'], signal.SIGSTOP)
            stopped = True
            import torch
            torch.set_num_threads(2)
            assert not torch.cuda.is_initialized(), 'Pause inspector must be CPU-only'
            checkpoint = run / 'latest.pth'
            with checkpoint.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
                stream.seek(0)
                saved = torch.load(stream, map_location='cpu', weights_only=False)
                assert os.fstat(stream.fileno()).st_ino == stat.st_ino
            assert saved['step'] >= config['minimum_checkpoint_step'] and saved['step'] % 1000 == 0
            assert saved['contract_sha256'] == config['hhc_contract_sha256']
            assert {'model', 'optimizer', 'scheduler', 'rng', 'step'} <= saved.keys()
            assert saved['optimizer']['state'] and {'python', 'numpy', 'torch', 'cuda'} <= saved['rng'].keys()
            accepted = {'checkpoint': str(checkpoint), 'bytes': stat.st_size, 'sha256': digest.hexdigest(),
                        'step': saved['step'], 'contract_sha256': saved['contract_sha256'],
                        'model_optimizer_scheduler_rng_present': True, 'cpu_only_inspection': True}
            write(control / 'saved_checkpoint_acceptance.json', accepted)
            # SIGTERM is pending while stopped; SIGCONT delivers it without restarting training.
            assert process(config['hhc_pid'], ['train_hhc_smpl_cfg_ablation.py', str(run)]) == identity
            os.kill(config['hhc_pid'], signal.SIGTERM)
            os.kill(config['hhc_pid'], signal.SIGCONT)
            stopped = False
            deadline = time.monotonic() + 120
            while not terminal(config['hhc_pid']) or not terminal(config['abc_pid']):
                if time.monotonic() > deadline:
                    raise RuntimeError('Observe original trainer/driver; do not restart or escalate signals blindly')
                time.sleep(1)
            queue = read(config['abc_status'])
            assert queue['phase_exit_codes'].get('A_historical') == -signal.SIGTERM
            manifest = read(run / 'training_manifest.json')
            manifest.update(status='paused_by_user_priority', step=accepted['step'],
                            intentional_pause={'reason': config['policy'], 'checkpoint_sha256': accepted['sha256'],
                                               'resume_only_after_intergen_complete_step': 704000})
            write(run / 'training_manifest.json', manifest)
            queue.update(status='paused_by_user_priority', intentional_pause=manifest['intentional_pause'])
            write(Path(config['abc_status']), queue)
            process(config['intergen_pid'], ['train_intergen_rotation.py', config['intergen_run']])
            gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            assert str(config['hhc_pid']) not in [line.split(',')[0].strip() for line in gpu.stdout.splitlines()]
            state.update(status='paused_checkpoint_verified_waiting_intergen', ended=time.time(),
                         saved_checkpoint=accepted, original_A_exit_code=-signal.SIGTERM,
                         original_driver_terminal=True, remote_gpu_processes=gpu.stdout.strip())
            write(control / 'STATUS.json', state)
        except BaseException as error:
            if stopped:
                os.kill(config['hhc_pid'], signal.SIGCONT)
            state.update(status='failed_needs_original_state_inspection', error=repr(error), ended=time.time())
            write(control / 'STATUS.json', state)
            raise


if __name__ == '__main__':
    main()
