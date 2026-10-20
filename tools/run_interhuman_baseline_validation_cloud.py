"""One cloud GPU baseline generation, followed by shared CPU scoring."""
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
CONTROL=ROOT/'output/interhuman_baseline_restore_20261016'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['intermask','intergen','timotion'],required=True)
    parser.add_argument('--wait-for-gpu',action='store_true')
    args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('GPU generation must run on the attributed cloud checkout')
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/'gpu_validation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|(0 if args.wait_for_gpu else fcntl.LOCK_NB))
        out=CONTROL/args.method
        out.mkdir(exist_ok=False)
        cache=out/'validation480.h5';report=out/'validation480_metrics.json'
        base=[sys.executable,'-u']
        if args.method=='intermask':
            generator='tools/ablate_intermask_generated_rotation.py'
            generation=base+[generator,'--cache',str(cache),'--limit','480','--frames','0','--device','cuda:0']
        elif args.method=='intergen':
            generator='tools/ablate_intergen_generated_rotation.py'
            generation=base+[generator,'--source',str(ROOT/'third_party/intergen_base_20261008'),
                '--checkpoint',str(ROOT/'third_party/intergen_base_20261008/checkpoints/intergen.ckpt'),
                '--cache',str(cache),'--limit','480','--frames','0','--device','cuda:0']
        else:
            generator='tools/export_timotion_interhuman_cloud.py'
            generation=base+[generator,'--output',str(cache),'--limit','480','--device','cuda:0']
        commands=[('generation',generation),('score',base+['tools/score_interhuman_raw_generation_cache.py','--method',args.method,'--cache',str(cache),'--output',str(report)])]
        state={'method':args.method,'status':'starting','pid':os.getpid(),'started':time.time(),
               'validation_only':True,'request_file_sha256':sha(ROOT/'output/interhuman_validation_requests/validation.json'),
               'sources':{p:sha(ROOT/p) for p in (generator,'tools/score_interhuman_raw_generation_cache.py','models/interhuman_evaluator_wrapper.py')},
               'stage_exit_codes':{},'local_checkpoint_downloads':False}
        status=out/'STATUS.json'
        for phase,command in commands:
            # Bind each child to the bytes it actually executes, even if a
            # queued runner predates a later compatibility repair.
            state.setdefault('stage_sources',{})[phase]={p:sha(ROOT/p) for p in (generator if phase=='generation' else 'tools/score_interhuman_raw_generation_cache.py','models/interhuman_evaluator_wrapper.py')}
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                     PYTHONPATH='/opt/hhc-baseline-deps',CUDA_VISIBLE_DEVICES='0' if phase=='generation' else '')
            if phase=='generation':
                free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                assert len(free.splitlines())==1 and int(free.strip())>=8192,'Insufficient GPU headroom; preserve existing training'
            with (out/f'{phase}.log').open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state.update(status='running',stage=phase,child_pid=child.pid);write(status,state)
                code=child.wait()
            state['stage_exit_codes'][phase]=code
            if code:
                state.update(status='failed',ended=time.time());write(status,state);return code
        state.update(status='complete',ended=time.time(),report=str(report),metrics=json.loads(report.read_text())['metrics'])
        write(status,state)
        print(json.dumps(state),flush=True)
        return 0


if __name__=='__main__':raise SystemExit(main())
