"""Rebind stopped CPU waiters to an actually restored official training launch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/HHC')
RUN = ROOT/'output/intergen_official_baseline_20261017'
REC = RUN/'recovery_20261020'
HHC = ROOT/'migration/hhc-after-official-intergen-resume.json'
EVAL = ROOT/'migration/official-intergen-eval20-after-final.json'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def atomic(path, value):
    temporary = Path(str(path)+'.recovery_20261020.tmp')
    write_new(temporary, value)
    os.replace(temporary, path)


def proc(pid):
    p = Path('/proc')/str(pid)
    try:
        s = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'birth': s[19], 'state': s[0],
                'command': (p/'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    launch = read(RUN/'LAUNCH.json')
    trainer = proc(launch['pid'])
    assert trainer and trainer['birth'] == launch['birth'] and trainer['state'] not in ('Z', 'X')
    events = Path(launch['command'][launch['command'].index('--event-dir')+1])
    proof = read(events/'first_resumed_batch.json')
    assert proof['global_step'] == proof['logging_counter'] == 255200 and proof['epoch'] == 725
    assert read(events/'full_state_restored.json')['optimizer_states'] == 196
    assert sha(HHC) == sha(REC/'hhc_config.before.json')
    assert sha(EVAL) == sha(REC/'eval_config.before.json')
    for pid in (21986, 25145, 26133):
        identity = proc(pid)
        assert not identity or identity['state'] in ('Z', 'X')
    write_new(REC/'waiter_rebind_intent.json', {'time_unix': time.time(), 'trainer': trainer})
    hhc, evaluation = read(HHC), read(EVAL)
    hhc['pins'][str(RUN/'LAUNCH.json')] = sha(RUN/'LAUNCH.json')
    for path in (ROOT/'tools/resume_official_intergen_full_state_cloud.py',
                 REC/'checkpoint_resume_acceptance.json', REC/'cpu_probe_v2/result.json',
                 REC/'input_restore_acceptance.json'):
        hhc['pins'][str(path)] = sha(path)
    atomic(HHC, hhc)
    evaluation['hhc_queue_config_sha256'] = sha(HHC)
    atomic(EVAL, evaluation)
    records = {}
    for name, source, config in [('hhc', 'resume_hhc_after_official_intergen_cloud.py', HHC),
                                  ('eval', 'run_official_intergen_reproduction_cloud.py', EVAL)]:
        command = [sys.executable, '-u', str(ROOT/'tools'/source), '--config', str(config),
                   '--recover-waiting', '--detach']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        write_new(REC/(name+'_waiter_recovery_execution.json'), {
            'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        assert result.returncode == 0, result.stderr
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
        identity = proc(parsed['pid'])
        assert identity and identity['state'] not in ('Z', 'X')
        records[name] = identity
    observer_log = REC/'observer.log'
    with observer_log.open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(ROOT/'tools/observe_official_intergen_baseline_cloud.py'), '--watch'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    records['observer'] = proc(child.pid)
    assert records['observer'] and records['observer']['state'] not in ('Z', 'X')
    atomic(RUN/'OBSERVER_LAUNCH.json', {'pid': child.pid, 'birth': records['observer']['birth'],
        'time_unix': time.time(), 'source': str(ROOT/'tools/observe_official_intergen_baseline_cloud.py'),
        'source_sha256': sha(ROOT/'tools/observe_official_intergen_baseline_cloud.py'),
        'log_path': str(observer_log), 'recovery': str(REC)})
    result = {'status': 'cpu_waiters_restored_pending_independent_readback',
              'trainer': trainer, 'identities': records, 'hhc_config_sha256': sha(HHC),
              'eval_config_sha256': sha(EVAL), 'launch_sha256': sha(RUN/'LAUNCH.json'),
              'hhc_still_saved_at': 4000, 'release_gate_unchanged': 'final704k plus actual20rep/MM plus independent review',
              'time_unix': time.time()}
    write_new(REC/'waiter_rebind_receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
