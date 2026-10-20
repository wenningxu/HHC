"""Suspend the attributed trainer just after an epoch checkpoint; keep it recoverable."""
from pathlib import Path
import json
import os
import signal
import time
import zipfile

ROOT=Path('/workspace/HHC')
RUN=ROOT/'output/intergen_official_baseline_20261017'
MIG=RUN/'edmond_handoff_20261020'

def proc(pid):
    p=Path('/proc')/str(pid)
    f=(p/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'birth':f[19],'state':f[0],
            'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}

def main():
    launch=json.loads((RUN/'LAUNCH.json').read_text())
    identity=proc(launch['pid'])
    assert identity['birth']==launch['birth'] and identity['state'] not in ('T','Z','X')
    assert 'resume_official_intergen_full_state_cloud.py' in identity['command']
    MIG.mkdir(exist_ok=True)
    with (MIG/'freeze_intent.json').open('x') as f:
        json.dump({'trainer':identity,'time':time.time()},f)
        f.flush();os.fsync(f.fileno())
    deadline=time.time()+600
    while time.time()<deadline:
        paths=list((RUN/'official_intergen_scratch_20261017/model').glob('*.ckpt'))
        if paths:
            checkpoint=max(paths,key=lambda p:p.stat().st_mtime_ns)
            age=time.time()-checkpoint.stat().st_mtime
            if 5<age<25 and zipfile.is_zipfile(checkpoint):
                now=proc(launch['pid'])
                assert now['birth']==launch['birth'] and now['state'] not in ('T','Z','X')
                os.kill(launch['pid'],signal.SIGSTOP)
                for _ in range(50):
                    identity=proc(launch['pid'])
                    if identity['state']=='T':break
                    time.sleep(.1)
                assert identity['state']=='T'
                snapshot=MIG/checkpoint.name
                os.link(checkpoint,snapshot)
                record={'status':'source_trainer_suspended_pending_destination_acceptance',
                        'trainer':identity,'snapshot':str(snapshot),'time':time.time(),
                        'source_launch':launch,'rollback':'SIGCONT only this same PID and birth if migration fails'}
                (MIG/'freeze_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps(record),flush=True)
                return
        time.sleep(2)
    raise RuntimeError('No fresh complete epoch checkpoint; trainer left running')

if __name__=='__main__':
    main()
