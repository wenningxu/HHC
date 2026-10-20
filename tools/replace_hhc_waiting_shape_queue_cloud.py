"""One-time guarded handoff: replace only the child-free, waiting PID 9549."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_hhc_train_shape_validation_v2_cloud import sha, write


def main():
    assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    previous = ROOT / 'output/hhc_train_shape_val_control_20261016'
    config_path = ROOT / 'migration/hhc-train-shape-validation-v2.json'
    config = json.loads(config_path.read_text())
    state_path = previous / 'STATUS.json'
    state = json.loads(state_path.read_text())
    assert state['pid'] == 9549 and state['status'] == 'waiting_for_zero_shape_scoring'
    assert state['phase_exit_codes'] == {} and 'child_pid' not in state
    assert state['config_sha256'] == 'c852d16bb747b2f6b37304515ffd83d377edf181c42521e2142ee3d7e8f17386'
    command = (Path('/proc/9549') / 'cmdline').read_bytes()
    assert b'run_hhc_train_shape_validation_cloud.py\0' in command
    assert b'/workspace/HHC/migration/hhc-train-shape-validation.json' in command
    assert json.loads((previous / 'launch_receipt.json').read_text())['pid'] == 9549
    assert not Path(config['control']).exists(), 'New queue already exists; never replay handoff'
    assert not (previous / 'superseded_by_v2.json').exists()
    assert not Path(config['output']).exists(), 'Old queue may have produced child output'
    assert {p.name for p in previous.iterdir()} == {'STATUS.json', 'driver.lock', 'launch.log', 'launch_receipt.json'}
    for name, digest in config['sources'].items():
        assert sha(ROOT / name) == digest, 'Bound source changed: ' + name
    audit_path = ROOT / 'output/hhc_sota_20261016/train_shape_queue_v2_cpu_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['queue_source_sha256'] == config['sources']['tools/run_hhc_train_shape_validation_v2_cloud.py']
    assert audit['config_sha256'] == sha(config_path) and len(audit['checks']) == 14
    assert audit['real_child_launches'] == 0
    evidence = {'status': 'verified_waiting_no_child_before_replacement', 'old_pid': 9549,
                'old_state': state, 'old_command': command.decode().replace('\0', ' '),
                'new_config_sha256': sha(config_path), 'cpu_audit_sha256': sha(audit_path),
                'started': time.time(), 'training_processes_to_preserve': [2353, 5690, 5729, 7178, 8321]}
    before = {pid: (Path('/proc') / str(pid) / 'cmdline').read_bytes() for pid in evidence['training_processes_to_preserve']}
    write(previous / 'superseded_by_v2.json', evidence)
    os.kill(9549, signal.SIGTERM)
    for _ in range(100):
        if not Path('/proc/9549').exists():
            break
        time.sleep(.1)
    assert not Path('/proc/9549').exists(), 'Original waiting driver not confirmed gone; stop handoff'
    state.update(status='superseded_before_any_child', ended=time.time(),
                 replacement_config=str(config_path), reason='Nested H5 JSON metadata parsing fix')
    write(state_path, state)
    run = subprocess.run([sys.executable, '-u', 'tools/run_hhc_train_shape_validation_v2_cloud.py',
                          '--config', str(config_path), '--detach'], cwd=ROOT, check=True, capture_output=True, text=True)
    receipt = json.loads((Path(config['control']) / 'launch_receipt.json').read_text())
    for _ in range(100):
        next_state_path = Path(config['control']) / 'STATUS.json'
        if next_state_path.exists():
            break
        time.sleep(.1)
    next_state = json.loads(next_state_path.read_text())
    pid = receipt['pid']
    assert next_state['pid'] == pid and next_state['status'] == 'waiting_for_zero_shape_scoring'
    assert next_state['phase_exit_codes'] == {} and 'child_pid' not in next_state
    assert b'run_hhc_train_shape_validation_v2_cloud.py' in (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    assert all((Path('/proc') / str(pid) / 'cmdline').read_bytes() == cmd for pid, cmd in before.items())
    evidence.update(status='replacement_confirmed_live_waiting', new_pid=pid, new_state=next_state,
                    original_waiting_pid_gone=True, termination_signal='SIGTERM',
                    original_exit_code=None, original_exit_code_scope='Not a child of this handoff; wait status unavailable',
                    preserved_process_commands_equal=True, ended=time.time())
    write(previous / 'superseded_by_v2.json', evidence)
    destination = ROOT / 'output/hhc_sota_20261016/train_shape_queue_v2_handoff.json'
    assert not destination.exists()
    write(destination, evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
