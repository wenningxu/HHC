"""Cloud VQ recovery acceptance (4 -> 6 updates), then the full rebuild.

Keeps the pre-existing InterGen trainer running. No local transfers, new pods,
or changes to the historical VQ training entrypoint.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud only')
    control=ROOT/'output/hhc_vq_rebuild_control_20261016';control.mkdir(exist_ok=True)
    with (control/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.run=args.run.resolve()
        if args.run.exists():raise FileExistsError('Existing run requires explicit recovery, never restart phases from zero')
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        assert len(free.splitlines())==1 and int(free.strip())>=8192
        state={'status':'running','pid':os.getpid(),'started':time.time(),'run':str(args.run),
            'origin':'fresh_cloud_rebuild_not_historical185k','preexisting_intergen_training_preserved':True,
            'local_checkpoint_downloads':False,'phase_exit_codes':{}}
        status=control/'STATUS.json';write(status,state)
        base=[sys.executable,'-u','tools/train_hhc_native_vq_cloud.py','--output',str(args.run),
              '--eval-device','cpu','--gpu-memory-fraction','.2','--wandb-mode','disabled']
        phases=[('initial4',base+['--stop-after-update','4']),
                ('resume6',base+['--resume','--stop-after-update','6']),('training',base+['--resume'])]
        for phase,command in phases:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH='/opt/hhc-vq-deps')
            with (control/f'{phase}.log').open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state.update(phase=phase,child_pid=child.pid);write(status,state);code=child.wait()
            state['phase_exit_codes'][phase]=code
            if code:
                state.update(status='failed',ended=time.time());write(status,state);return code
            if phase in ('initial4','resume6'):
                manifest=json.loads((args.run/'training_manifest.json').read_text())
                assert manifest['status']=='checkpoint_stop' and manifest['last_update']==(4 if phase=='initial4' else 6)
                write(control/f'{phase}_acceptance.json',manifest)
        state.update(status='complete',ended=time.time());write(status,state);return 0


if __name__=='__main__':raise SystemExit(main())
