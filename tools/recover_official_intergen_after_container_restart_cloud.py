"""One-shot recovery after verified process loss, preserving all old evidence."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/HHC')
RUN = ROOT/'output/intergen_official_baseline_20261017'
RECOVERY = RUN/'recovery_20261020'
HHC = ROOT/'migration/hhc-after-official-intergen-resume.json'
EVAL = ROOT/'migration/official-intergen-eval20-after-final.json'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def proc(pid):
    p = Path('/proc')/str(pid)
    try:
        values = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'birth': values[19], 'state': values[0],
                'argv': [x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    assert Path(__file__).resolve().parents[1] == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    lock = (RECOVERY/'transaction.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    intent = RECOVERY/'training.launch_intent.json'
    assert not intent.exists(), 'A launch intent exists: inspect it; never replay blindly'
    old = read(RUN/'LAUNCH.json')
    for pid in (old['pid'], 21986, 25145, 26133):
        actual = proc(pid)
        assert not actual or actual['state'] in ('Z', 'X'), (pid, actual)
    gpu_names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    assert gpu_names == ['NVIDIA GeForce RTX 4090'], gpu_names
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not apps.strip()
    hhc, evaluation = read(HHC), read(EVAL)
    for path, expected in {**hhc['pins'], **evaluation['pins']}.items():
        assert sha(path) == expected, path
    catalogue = read(hhc['official_source_catalogue'])
    source = Path(old['cwd'])
    for relative, expected in catalogue['tracked_files'].items():
        expected = old.get('source_changes', {}).get(relative, {}).get('after', expected)
        assert sha(source/relative) == expected, relative
    for directory in (Path(hhc['control']), Path(evaluation['control'])):
        state = read(directory/'STATUS.json')
        assert state['status'] == 'waiting_for_official_training'
        assert not state.get('phase') and not state.get('child_pid') and not state.get('phase_exit_codes')
        assert not state.get('dispatch_started') and 'child_exit_code' not in state
        assert not list(directory.glob('*intent*'))
    check = read(RECOVERY/'cpu_probe/result.json')
    assert check['status'] == 'passed' and not check['cpu_cuda_initialized']
    wrapper = ROOT/'tools/resume_official_intergen_full_state_cloud.py'
    assert sha(wrapper) == '7d3ee8e62a7df2a71eb9de3114f8113f92db23faa7c514ed20b0b589765049e3'
    candidate = read(RECOVERY/'checkpoint_candidate_cpu.json')
    full = Path(candidate['snapshot'])
    assert candidate['step'] == 255200 and candidate['optimizer_states'] == 196
    assert full.stat().st_size == candidate['bytes'] and sha(full) == candidate['sha256']
    assert zipfile.is_zipfile(full)
    paths = {'LAUNCH.before.json': RUN/'LAUNCH.json', 'hhc_config.before.json': HHC,
             'eval_config.before.json': EVAL, 'hhc_STATUS.before.json': Path(hhc['control'])/'STATUS.json',
             'eval_STATUS.before.json': Path(evaluation['control'])/'STATUS.json',
             'OBSERVER_STATUS.before.json': RUN/'OBSERVER_STATUS.json',
             'OBSERVER_LAUNCH.before.json': RUN/'OBSERVER_LAUNCH.json'}
    archives = {}
    for name, path in paths.items():
        data = path.read_bytes()
        with (RECOVERY/name).open('xb') as handle:
            handle.write(data)
        assert (RECOVERY/name).read_bytes() == data
        archives[name] = {'sha256': sha(path), 'bytes': len(data)}
    snapshot = RUN/'accepted_checkpoints'/full.name
    assert not snapshot.exists()
    os.link(full, snapshot)
    assert snapshot.stat().st_ino == full.stat().st_ino
    partial = full.with_name('epoch=725-step=255552.ckpt')
    assert partial.is_file() and not zipfile.is_zipfile(partial)
    partial_record = {'original': str(partial), 'bytes': partial.stat().st_size, 'sha256': sha(partial)}
    quarantine = RECOVERY/'interrupted_checkpoints'
    quarantine.mkdir()
    destination = quarantine/partial.name
    partial.rename(destination)
    assert sha(destination) == partial_record['sha256']
    partial_record['preserved_at'] = str(destination)
    candidate['original_checkpoint'] = str(full)
    candidate['snapshot'] = str(snapshot)
    candidate['source_catalogue_verified_files'] = len(catalogue['tracked_files'])
    accepted = RECOVERY/'checkpoint_resume_acceptance.json'
    write_new(accepted, candidate)
    events = RECOVERY/'training_events'
    events.mkdir()
    command = ['/workspace/intergen-official-env-py38/bin/python', '-u', str(wrapper),
               '--runtime-wrapper', str(ROOT/'tools/run_official_intergen_checkpointed_cloud.py'),
               '--checkpoint-receipt', str(accepted), '--event-dir', str(events)]
    write_new(intent, {'command': command, 'cwd': old['cwd'], 'checkpoint_sha256': candidate['sha256'],
                      'resume_step': candidate['step'], 'wrapper_sha256': sha(wrapper), 'time_unix': time.time()})
    descriptor = os.open(RECOVERY, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    log_path = RUN/'train.recovery_20261020.log'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MPLBACKEND': 'Agg'}
    with log_path.open('x') as log:
        child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = proc(child.pid)
    assert identity and identity['state'] not in ('Z', 'X')
    launch = dict(old)
    launch.update(pid=child.pid, birth=identity['birth'], command=command, time=time.time(),
                  log_path=str(log_path), previous_launch=str(RECOVERY/'LAUNCH.before.json'))
    launch['runtime_changes'] = dict(old['runtime_changes'], full_state_resume={
        'wrapper': str(wrapper), 'wrapper_sha256': sha(wrapper), 'resume_step': 255200,
        'checkpoint_receipt': str(accepted), 'checkpoint_receipt_sha256': sha(accepted),
        'checkpoint_sha256': candidate['sha256'],
        'pending_official_epoch_scheduler_hook_replayed_once': True,
        'rng_dataloader_bitwise_equivalence_claimed': False})
    temporary = RUN/'LAUNCH.recovery.tmp'
    write_new(temporary, launch)
    os.replace(temporary, RUN/'LAUNCH.json')
    result = {'status': 'recovery_dispatched_pending_actual_update', 'trainer': identity,
              'launch_sha256': sha(RUN/'LAUNCH.json'), 'log_path': str(log_path),
              'resume_checkpoint': candidate, 'partial_checkpoint': partial_record,
              'archives': archives, 'source_catalogue_verified_files': len(catalogue['tracked_files']),
              'cpu_probe_sha256': sha(RECOVERY/'cpu_probe/result.json'),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'hhc_not_launched': True, 'time_unix': time.time()}
    write_new(RECOVERY/'dispatch_receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
