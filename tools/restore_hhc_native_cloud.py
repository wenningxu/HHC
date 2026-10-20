"""Cloud-only, CPU reconstruction of HHC native inputs; never downloads weights."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'output/hhc_native_restore_control_20261016'
PILOT=ROOT/'dataset/interhuman_smpl_native_restore_pilot_20261016'
FULL=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['pilot','full'])
    args=parser.parse_args()
    if str(ROOT) != '/workspace/HHC':
        raise RuntimeError('This launcher is restricted to the attributed cloud checkout /workspace/HHC')
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected='4df6ded9ffabbea3b3d755f2acea1329dea166b7a143218d354612bfc76bef65'
        source=ROOT/'output/interhuman_raw_download_20261003.json'
        if sha(source)!=expected:raise RuntimeError('Pinned raw download manifest changed')
        target=PILOT if args.phase=='pilot' else FULL
        if args.phase=='full':
            pilot=json.loads((PILOT/'manifest.json').read_text())
            assert pilot['summary']['status']=='pilot_complete' and pilot['summary']['files']==32
            assert pilot['summary']['raw_arrays_bitwise_restored']==256
        if target.exists():raise RuntimeError(f'Output already exists; inspect its status, do not restart or overwrite: {target}')
        command=[sys.executable,'-u',str(ROOT/'tools/build_interhuman_smpl_pair.py'),'--output',str(target)]
        if args.phase=='pilot':command+=['--limit','32']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        state={'phase':args.phase,'status':'starting','wrapper_pid':os.getpid(),'target':str(target),
               'started':time.time(),'compute':'cloud_cpu_only','gpu_visible':False,
               'checkpoint_downloads':False,'raw_manifest_sha256':expected,
               'sources':{name:sha(ROOT/name) for name in ('tools/build_interhuman_smpl_pair.py','dataset/interhuman_smpl_pair.py','tools/reconstruct_interhuman_raw_body.py','dataset/interhuman_hhc_adapter.py')}}
        status=CONTROL/f'{args.phase}_status.json'
        with (CONTROL/f'{args.phase}.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',builder_pid=process.pid);write(status,state)
            code=process.wait()
        state.update(status='complete' if code==0 else 'failed',exit_code=code,ended=time.time())
        if code==0:
            manifest=json.loads((target/'manifest.json').read_text())
            state['summary']=manifest['summary']
            state['manifest_sha256']=sha(target/'manifest.json')
            state['motions_h5_sha256']=sha(target/'motions.h5')
        write(status,state)
        print(json.dumps(state),flush=True)
        return code


if __name__=='__main__':
    raise SystemExit(main())
