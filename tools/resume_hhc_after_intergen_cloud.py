"""Continue saved HHC A/B/C only after the user's InterGen704k priority finishes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pause_hhc_for_intergen_cloud import read, write, sha, sources, terminal

ROOT = Path(__file__).resolve().parents[1]
WAITING = {'waiting_for_hhc_saved_pause', 'waiting_for_intergen_complete', 'waiting_for_gpu_headroom'}


def completion_gate(status, identity, expected, live_trainers):
    return (status.get('status') == 'complete' and status.get('step') == 704000
            and all(identity.get(key) == value for key, value in expected.items())
            and not live_trainers)


def intergen_processes(run):
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes()
            if b'train_intergen_rotation.py' in command and str(run).encode() in command:
                if not terminal(int(path.parent.name)):
                    found.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def checkpoint_acceptance(path, step, contract=None):
    import hashlib
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized(), 'Waiter must remain CPU-only'
    with Path(path).open('rb') as stream:
        stat = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        assert os.fstat(stream.fileno()).st_ino == stat.st_ino
    assert saved['step'] == step
    if contract:
        assert saved['contract_sha256'] == contract
        assert saved['optimizer']['state'] and {'python', 'numpy', 'torch', 'cuda'} <= saved['rng'].keys()
    return {'path': str(path), 'step': step, 'bytes': stat.st_size, 'sha256': digest.hexdigest(),
            'cpu_only': True, 'contract_sha256': contract}


def arm_command(arm, spec, resume=False):
    command = [sys.executable, '-u', 'tools/train_hhc_smpl_cfg_ablation.py', '--vq-spec', spec,
               '--output', arm['output'], '--conditioning-arm', arm['conditioning'],
               '--token-window', arm['window'], '--steps', '40000', '--microbatch', '4',
               '--seed', '114514', '--eval-device', 'cpu', '--gpu-memory-fraction', '.5',
               '--wandb-mode', 'disabled']
    return command + (['--resume'] if resume else [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--recover-waiting', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    assert str(ROOT) == '/workspace/HHC'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Parent never owns a CUDA device'
    config = read(args.config)
    policy = read(config['priority_config'])
    assert sha(config['priority_config']) == config['priority_config_sha256']
    assert policy['policy'] == 'intergen_exclusive_until_complete704000'
    assert sha(ROOT / 'tools/pause_hhc_for_intergen_cloud.py') == config['pause_source_sha256']
    source_sha = sha(Path(__file__))
    sources(policy)
    assert sha(policy['vq_spec']) == policy['vq_spec_sha256']
    abc = read(config['original_abc_config'])
    assert sha(config['original_abc_config']) == config['original_abc_config_sha256']
    run = Path(policy['intergen_run'])
    control = Path(config['control'])
    status_path = control / 'STATUS.json'
    if args.check_only:
        identity = read(run / 'config.json')
        status = read(run / 'STATUS.json')
        print(json.dumps({'status': 'preflight_passed_no_child_launch',
                          'intergen_status': status.get('status'), 'intergen_step': status.get('step'),
                          'release_gate': completion_gate(status, identity, config['intergen_identity'],
                                                         intergen_processes(run)),
                          'A_resume_command': arm_command(abc['arms'][0], policy['vq_spec'], True)}))
        return
    control.mkdir(exist_ok=True)
    if args.detach:
        if status_path.exists():
            prior = read(status_path)
            assert args.recover_waiting and prior['status'] in WAITING and terminal(prior['pid'])
            assert not prior.get('phase_exit_codes') and not prior.get('phase'), 'Only recover a waiter before any arm ran'
        name = f'recovery_launch_{int(time.time())}.json' if args.recover_waiting else 'launch_receipt.json'
        with (control / name).open('x') as receipt, (control / 'launch.log').open('a') as log:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(args.config.resolve())]
            if args.recover_waiting:
                command.append('--recover-waiting')
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            json.dump({'pid': child.pid, 'config_sha256': sha(args.config), 'started': time.time()}, receipt)
        print(json.dumps({'pid': child.pid, 'status': 'launched_intergen_priority_waiter'}))
        return
    with (control / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read(status_path) if status_path.exists() else None
        if prior:
            assert args.recover_waiting and prior['status'] in WAITING and terminal(prior['pid'])
            assert not prior.get('phase_exit_codes') and not prior.get('phase'), 'Running/completed phases need explicit recovery'
        state = {'status': 'waiting_for_hhc_saved_pause', 'pid': os.getpid(), 'started': time.time(),
                 'config_sha256': sha(args.config), 'dispatcher_source_sha256': source_sha,
                 'policy': policy['policy'], 'phase_exit_codes': {}, 'local_checkpoint_downloads': False,
                 'previous_waiter_pid': prior['pid'] if prior else None}
        write(status_path, state)

        def unchanged():
            assert sha(Path(__file__)) == source_sha
            assert sha(config['priority_config']) == config['priority_config_sha256']
            assert sha(ROOT / 'tools/pause_hhc_for_intergen_cloud.py') == config['pause_source_sha256']
            assert sha(config['original_abc_config']) == config['original_abc_config_sha256']
            assert sha(ROOT / 'tools/train_intergen_rotation.py') == config['intergen_identity']['trainer_source_sha256']
            sources(policy)
            assert sha(policy['vq_spec']) == policy['vq_spec_sha256']

        def gate():
            try:
                identity = read(run / 'config.json')
                status = read(run / 'STATUS.json')
            except (FileNotFoundError, json.JSONDecodeError):
                return False, {'status': 'unavailable_or_transient_invalid_json'}
            ready = completion_gate(status, identity, config['intergen_identity'], intergen_processes(run))
            return ready, {'status': status.get('status'), 'step': status.get('step')}

        try:
            pause_path = Path(policy['pause_control']) / 'STATUS.json'
            while True:
                unchanged()
                if pause_path.exists():
                    pause = read(pause_path)
                    if pause['status'] == 'paused_checkpoint_verified_waiting_intergen':
                        break
                    assert pause['status'] == 'waiting_for_safe_checkpoint', pause['status']
                time.sleep(30)
            assert terminal(policy['hhc_pid']) and terminal(policy['abc_pid'])
            accepted = pause['saved_checkpoint']
            assert accepted['step'] >= policy['minimum_checkpoint_step']
            state.update(status='waiting_for_intergen_complete', saved_hhc_step=accepted['step'],
                         saved_hhc_checkpoint_sha256=accepted['sha256'])
            write(status_path, state)
            while True:
                unchanged()
                ready, observed = gate()
                state['last_intergen_observation'] = observed
                write(status_path, state)
                if ready:
                    break
                time.sleep(600)
            final = checkpoint_acceptance(run / 'latest.pth', 704000)
            write(control / 'intergen_complete_checkpoint_acceptance.json', final)
            actual = checkpoint_acceptance(accepted['checkpoint'], accepted['step'], policy['hhc_contract_sha256'])
            assert actual['sha256'] == accepted['sha256'] and actual['bytes'] == accepted['bytes']
            write(control / 'saved_hhc_checkpoint_acceptance.json', actual)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH='/opt/hhc-vq-deps',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            for index, arm in enumerate(abc['arms']):
                unchanged()
                ready, _ = gate()
                assert ready, 'InterGen priority completion evidence changed'
                out = Path(arm['output'])
                if index == 0:
                    manifest = read(out / 'training_manifest.json')
                    assert manifest['status'] == 'paused_by_user_priority'
                    assert manifest['step'] == accepted['step']
                    assert manifest['contract_sha256'] == policy['hhc_contract_sha256']
                else:
                    assert not out.exists(), 'Never replay an existing B/C output'
                while True:
                    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                                    '--format=csv,noheader,nounits'], text=True).splitlines()
                    assert len(free) == 1
                    if int(free[0]) >= 13 * 1024:
                        break
                    state['status'] = 'waiting_for_gpu_headroom'
                    write(status_path, state)
                    time.sleep(600)
                    unchanged()
                    assert gate()[0]
                with (control / f'{arm["name"]}.log').open('x') as log:
                    child = subprocess.Popen(arm_command(arm, policy['vq_spec'], index == 0),
                                             cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    state.update(status='running', phase=arm['name'], child_pid=child.pid)
                    write(status_path, state)
                    code = child.wait()
                state['phase_exit_codes'][arm['name']] = code
                write(status_path, state)
                assert code == 0, 'No automatic replay after training failure'
                manifest = read(out / 'training_manifest.json')
                assert manifest['status'] == 'pilot_complete_validation_only' and manifest['step'] == 40000
                assert manifest['contract']['vq_spec_sha256'] == policy['vq_spec_sha256']
                assert manifest['contract']['steps'] == 40000 and manifest['contract']['seed'] == 114514
            state.update(status='complete_validation_training_only', ended=time.time())
            write(status_path, state)
        except BaseException as error:
            state.update(status='failed_needs_explicit_phase_recovery', error=repr(error), ended=time.time())
            write(status_path, state)
            raise


if __name__ == '__main__':
    main()
