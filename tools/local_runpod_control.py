"""Local ownership, status and resumable recovery collection for HHC Runpod."""
import hashlib,json,msvcrt,os,pathlib,subprocess,time
from local_compute_control import atomic_json,event,digest,checkpoint_downloads_allowed
ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTROL=ROOT/'migration/control'
RUN='/workspace/HHC/output/intergen_rotation_scratch_runpod_20261015'
SSH=['ssh','-i',r'C:\Users\111\.ssh\id_ed25519_newstation','-p','33923','-o','BatchMode=yes','-o','ConnectTimeout=20','root@157.157.221.29']
def remote(code):
    result=subprocess.run(SSH+['python -'],input=code,text=True,capture_output=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout)
def cycle():
    if not (CONTROL/'old-compute-fenced.json').exists():
        code="""import subprocess,os,signal,time,json
prefix='python -u /dev/shm/hhc_local_control_20261015/tools/train_intergen_rotation.py --scratch --resume --output /dev/shm/hhc_local_control_20261015/output/intergen_rotation_scratch_20261010'
def owned():
 rows=subprocess.getoutput('ps -eo pid=,args=').splitlines()
 return [int(r.strip().split(None,1)[0]) for r in rows if len(r.strip().split(None,1))==2 and r.strip().split(None,1)[1].startswith(prefix)]
for pid in owned():os.kill(pid,signal.SIGTERM)
subprocess.run(['tmux','kill-session','-t','hhc_local_scratch_20261015'],capture_output=True)
time.sleep(3)
assert not owned(), 'Old attributed trainer still alive'
print(json.dumps({'old_attributed_training_stopped':True,'time':time.time()}))
"""
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','newstation','python3 -'],input=code,text=True,capture_output=True,timeout=45)
        if result.returncode==0:atomic_json(CONTROL/'old-compute-fenced.json',json.loads(result.stdout))
    current=remote(f'''import json,pathlib,subprocess
r=pathlib.Path({RUN!r})
def read(n):
 try:return json.loads((r/n).read_text())
 except FileNotFoundError:return None
print(json.dumps({{'status':read('STATUS.json'),'recovery':read('RECOVERY.json'),'processes':subprocess.getoutput("pgrep -af '[t]rain_intergen_rotation.py'")}}))
''')
    atomic_json(CONTROL/'runpod-status.json',dict(current,checked_at=time.time()))
    if not current['processes'] and current['recovery'] and (current['status'] or {}).get('status')!='complete':
        command='cd /workspace/HHC && nohup bash tools/runpod_bootstrap.sh >> /workspace/hhc-training.log 2>&1 < /dev/null &'
        subprocess.run(SSH+[command],capture_output=True,timeout=30,check=True)
        event('runpod_resume_dispatched',pod='3aj1czzq0a9x64')
    if not checkpoint_downloads_allowed():return
    pending=CONTROL/'runpod-transfer.json'
    if pending.exists():job=json.loads(pending.read_text())
    elif current['recovery']:
        rec=current['recovery'];step=rec['step']
        destination=ROOT/f'checkpoints/local_backups/intergen_scratch_runpod_20261015_step{step:06d}/recovery.pt'
        receipt=destination.with_suffix('.pt.receipt.json')
        if receipt.exists():return
        path=RUN+f'/transfer_step{step:06d}.pt'
        job=remote(f'''import pathlib,json,hashlib,os
r=pathlib.Path({RUN!r}); p=pathlib.Path({path!r})
if not p.exists():os.link(r/'recovery.pt',p)
with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps({{'step':{step},'remote':str(p),'bytes':p.stat().st_size,'sha256':sha}}))''')
        # A save racing the receipt is rejected rather than mislabelling its step.
        if job['sha256']!=rec['sha256']:raise RuntimeError('Recovery changed while snapshotting; retain and inspect snapshot')
        atomic_json(pending,job)
    else:return
    destination=ROOT/f"checkpoints/local_backups/intergen_scratch_runpod_20261015_step{job['step']:06d}/recovery.pt"
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_suffix('.pt.partial')
    if not destination.exists():
        command=['sftp','-i',r'C:\Users\111\.ssh\id_ed25519_newstation','-P','33923','-o','BatchMode=yes','-o','ConnectTimeout=20','-b','-','root@157.157.221.29']
        result=subprocess.run(command,input=f'reget "{job["remote"]}" "{partial.as_posix()}"\n',text=True,timeout=7200)
        if result.returncode:raise RuntimeError('Runpod SFTP failed')
        assert partial.stat().st_size==job['bytes'] and digest(partial)==job['sha256']
        os.replace(partial,destination)
    assert destination.stat().st_size==job['bytes'] and digest(destination)==job['sha256']
    atomic_json(destination.with_suffix('.pt.receipt.json'),dict(job,file=str(destination),verified_at=time.time(),cloud_status='pending'))
    remote(f'import pathlib,json; pathlib.Path({job["remote"]!r}).unlink(missing_ok=True); print(json.dumps({{"transfer_link_released":True}}))')
    pending.unlink();event('runpod_recovery_verified',**job)
def main():
    with (CONTROL/'runpod-control.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        atomic_json(CONTROL/'runpod-control.pid.json',{'pid':os.getpid(),'started':time.time()})
        while True:
            owner=json.loads((CONTROL/'compute-owner.json').read_text())
            if owner.get('backend')!='runpod':return
            try:cycle()
            except Exception as exc:event('runpod_control_error',error=str(exc))
            time.sleep(3600)
if __name__=='__main__':main()
