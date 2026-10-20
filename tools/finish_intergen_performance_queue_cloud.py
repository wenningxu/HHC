"""Finish CPU queue bookkeeping after verified fast InterGen launch."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from intergen_speed_probe_cloud import proc, same_live, write
from pause_hhc_for_intergen_cloud import sources

ROOT = Path(__file__).resolve().parents[1]
assert str(ROOT) == '/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
control = ROOT / 'output/intergen_fast_transition_control_20261017'
guard = json.loads((control / 'guard.json').read_bytes())
assert not same_live(guard['coordinator']), 'Never race an active coordinator'
run = ROOT / 'output/intergen_rotation_scratch_runpod_20261015'
cfg = json.loads((ROOT / 'migration/hhc-after-fast-intergen-resume.json').read_bytes())
actual = json.loads((run / 'config.json').read_bytes())
assert all(actual.get(k) == v for k, v in cfg['intergen_identity'].items())
assert json.loads((run / 'STATUS.json').read_bytes())['step'] >= 84100
trainer = proc(20042)
assert trainer and trainer['state'] not in ('Z', 'X', 'T', 't')
assert 'tools/train_intergen_rotation_fast_cloud.py ' in trainer['command'] and str(run) in trainer['command']
assert hashlib.sha256((ROOT / 'tools/train_intergen_rotation_fast_cloud.py').read_bytes()).hexdigest() == actual['trainer_source_sha256']
sources(json.loads((ROOT / 'migration/hhc-compute-priority.json').read_bytes()))
old_path = ROOT / 'output/hhc_after_intergen_resume_control_20261017/STATUS.json'
old = json.loads(old_path.read_bytes())
assert old['status'] == 'waiting_for_intergen_complete' and old['pid'] == 17952
assert not old.get('phase') and not old.get('phase_exit_codes')
identity = proc(17952)
assert identity and 'resume_hhc_after_intergen_cloud.py' in identity['command']
os.kill(17952, signal.SIGTERM)
deadline = time.monotonic()+10
while same_live(identity):
    assert time.monotonic() < deadline
    time.sleep(.1)
old.update(status='superseded_by_performance_waiter',
           superseding_control='/workspace/HHC/output/hhc_after_intergen_fast_resume_control_20261017')
write(old_path, old)
waiter = ROOT / 'tools/resume_hhc_after_intergen_fast_cloud.py'
config = ROOT / 'migration/hhc-after-fast-intergen-resume.json'
for flags in (['--check-only'], ['--detach']):
    result = subprocess.run([sys.executable, str(waiter), '--config', str(config), *flags],
                            capture_output=True, text=True, check=True, timeout=30)
    write(control / ('waiter_preflight.json' if flags[0] == '--check-only' else 'waiter_launch.json'),
          {'stdout': result.stdout, 'exit_code': result.returncode})
deadline = time.monotonic()+30
while True:
    p = ROOT / 'output/hhc_after_intergen_fast_resume_control_20261017/STATUS.json'
    if p.exists():
        waiting = json.loads(p.read_bytes())
        identity = proc(waiting['pid'])
        if waiting['status'] == 'waiting_for_intergen_complete' and identity and same_live(identity):
            assert waiting['saved_hhc_step'] == 4000 and not waiting['phase_exit_codes']
            break
    assert time.monotonic() < deadline
    time.sleep(.2)
status = {'status': 'performance_continuation_active', 'resume_step': 84000,
          'trainer': proc(20042), 'hhc_waiter_pid': waiting['pid'], 'saved_hhc_step': 4000,
          'micro_batch': 16, 'accumulation': 2, 'effective_batch': 32,
          'original_coordinator_absent_after_launch': True,
          'cpu_queue_completion_by_explicit_repair': True}
write(control / 'STATUS.json', status)
write(control / 'coordinator_done.json', {'status': 'finished_by_cpu_queue_repair', 'unix': time.time()})
print(json.dumps(status), flush=True)
