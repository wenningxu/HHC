"""Local ownership of HHC compute status and resumable artifact downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import msvcrt

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/'migration/control'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', 'newstation']
STAGE = '/dev/shm/hhc_local_control_20261015'
RUN = STAGE+'/output/intergen_rotation_scratch_20261010'
ORIGINAL = '/mnt/8T-HDD/wenning/HHC/output/intergen_rotation_scratch_20261010'

def event(kind, **details):
    CONTROL.mkdir(parents=True, exist_ok=True)
    row = {'time_unix': time.time(), 'event': kind, **details}
    with (CONTROL/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)

def checkpoint_downloads_allowed():
    policy=CONTROL/'local-checkpoint-policy.json'
    if not policy.exists(): return True
    return json.loads(policy.read_text(encoding='utf-8')).get('checkpoint_downloads_enabled',False) is True

def download(remote, destination, expected_hash, expected_bytes):
    if not checkpoint_downloads_allowed():
        raise RuntimeError('Local checkpoint download disabled by user policy')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_bytes and digest(destination) == expected_hash:
        return destination
    partial = destination.with_suffix(destination.suffix+'.partial')
    if partial.exists() and partial.stat().st_size > expected_bytes:
        raise RuntimeError('Partial file exceeds expected artifact size')
    batch = f'reget "{remote}" "{partial.as_posix()}"\n'
    result = subprocess.run(['sftp','-b','-','-o','BatchMode=yes','-o','ConnectTimeout=20',
                             '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','newstation'],
                            input=batch.encode(), timeout=3600)
    if result.returncode:
        raise RuntimeError(f'SFTP failed: {result.returncode}')
    if partial.stat().st_size != expected_bytes or digest(partial) != expected_hash:
        raise RuntimeError('Downloaded artifact did not pass size/SHA256 checks')
    os.replace(partial, destination)
    receipt = {'file':str(destination),'remote':remote,'sha256':expected_hash,
               'bytes':expected_bytes,'verified_at':time.time(),'cloud_status':'pending'}
    atomic_json(destination.with_suffix(destination.suffix+'.receipt.json'), receipt)
    event('artifact_verified', **receipt)
    return destination

def inspect_remote():
    script = r'''
import pathlib,json,subprocess
stage=pathlib.Path('/dev/shm/hhc_local_control_20261015')
run=stage/'output/intergen_rotation_scratch_20261010'
def read(name):
 try: return json.loads((run/name).read_text())
 except (OSError,ValueError): return None
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
trainers=[x for x in processes if '/dev/shm/hhc_local_control_20261015/tools/train_intergen_rotation.py' in x and '--check-resume-only' not in x and 'python ' in x]
metrics=[]
for p in sorted(run.glob('validation_*/validation_metrics.json')):
 try: metrics.append({'path':str(p),'metrics':json.loads(p.read_text())['arms']['raw']})
 except (OSError,ValueError,KeyError): pass
print(json.dumps({'stage_exists':stage.exists(),'status':read('STATUS.json'),'recovery':read('RECOVERY.json'),'trainers':trainers,'validations':metrics}))
'''
    result = subprocess.run(SSH+['python3 -'],input=script,text=True,capture_output=True,timeout=120)
    if result.returncode: raise RuntimeError(f'SSH failed {result.returncode}: {result.stderr[-500:]}')
    value = json.loads(result.stdout)
    atomic_json(CONTROL/'remote-status.json',value)
    return value

def monitor(once=False):
    while True:
        owner=CONTROL/'compute-owner.json'
        if owner.exists() and json.loads(owner.read_text()).get('backend')=='runpod':
            event('old_server_monitor_disabled',reason='Compute owner switched to Runpod')
            return
        try:
            current = inspect_remote()
            event('hourly_status', **current)
            # Restart only our attributed tmux session and an existing staged run.
            status = current.get('status') or {}
            if current['stage_exists'] and not current['trainers'] and status.get('status') != 'complete':
                launch = ROOT/'migration/launch_remote.sh'
                result = subprocess.run(SSH+['bash -s'],input=launch.read_bytes(),capture_output=True,timeout=120)
                event('resume_dispatched',exit_code=result.returncode,output=result.stdout.decode(errors='replace')[-1000:])
            elif not current['stage_exists']:
                event('stage_missing_requires_redeploy',fallback='Runpod if server unavailable')
        except Exception as exc:
            event('monitor_error',error=str(exc))
        if once: return
        time.sleep(3600)

def initial_backup():
    if not checkpoint_downloads_allowed():
        event('checkpoint_download_disabled_by_user_policy',mode='initial-backup')
        return
    destination=ROOT/'checkpoints/local_backups/intergen_scratch_20261010_step000000/recovery.pt'
    while True:
        try:
            download(ORIGINAL+'/recovery.pt',destination,
                     '100d13d3a3e014647688f074cdfa704f4ef3e524fcdf8d591fe71783ce4a6116',729659053)
            break
        except Exception as exc:
            event('initial_download_retry',error=str(exc))
            time.sleep(60)
    while True:
        try:
            pending_path=CONTROL/'rolling-transfer.json'
            if not checkpoint_downloads_allowed():
                event('checkpoint_download_disabled_by_user_policy',mode='rolling-backup')
                return
            current=None
            job=json.loads(pending_path.read_text()) if pending_path.exists() else None
            if job is None:
                current=inspect_remote()
                receipt=current.get('recovery')
            else:
                receipt=None
            if job is None and receipt:
                step=int(receipt['step'])
                destination=ROOT/f'checkpoints/local_backups/intergen_scratch_20261010_step{step:06d}/recovery.pt'
                if not destination.exists():
                    snapshot=RUN+f'/transfer_step{step:06d}.pt'
                    script=f'''import os,pathlib,hashlib,json
source=pathlib.Path({(RUN+'/recovery.pt')!r})
snapshot=pathlib.Path({snapshot!r})
if not snapshot.exists(): os.link(source,snapshot)
with snapshot.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
if actual!={receipt['sha256']!r}:
 snapshot.unlink()
 raise RuntimeError('Checkpoint advanced before snapshot; retry next cycle')
print(json.dumps({{'sha256':actual,'bytes':snapshot.stat().st_size}}))
'''
                    result=subprocess.run(SSH+['python3 -'],input=script,text=True,capture_output=True,timeout=120)
                    if result.returncode: raise RuntimeError(result.stderr[-500:])
                    info=json.loads(result.stdout)
                    job={'step':step,'remote':snapshot,'bytes':info['bytes'],'sha256':info['sha256']}
                    atomic_json(pending_path,job)
            if job is not None:
                step=int(job['step'])
                assert job['remote']==RUN+f'/transfer_step{step:06d}.pt'
                destination=ROOT/f'checkpoints/local_backups/intergen_scratch_20261010_step{step:06d}/recovery.pt'
                download(job['remote'],destination,job['sha256'],job['bytes'])
                cleanup=f'import pathlib; pathlib.Path({job["remote"]!r}).unlink()'
                subprocess.run(SSH+['python3 -'],input=cleanup,text=True,timeout=120,check=True)
                pending_path.unlink()
            if current and (current.get('status') or {}).get('status')=='complete':
                event('training_complete_local_backup_finished',status=current['status'])
                return
        except Exception as exc:
            event('rolling_backup_error',error=str(exc))
        time.sleep(60 if (CONTROL/'rolling-transfer.json').exists() else 3600)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['monitor','inspect','initial-backup'])
    args=parser.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    with (CONTROL/(args.mode+'.lock')).open('a+b') as lock:
        lock.seek(0)
        if lock.read(1)==b'': lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        atomic_json(CONTROL/(args.mode+'.pid.json'),{'pid':os.getpid(),'started':time.time()})
        if args.mode=='initial-backup': initial_backup()
        else: monitor(once=args.mode=='inspect')

if __name__=='__main__': main()
