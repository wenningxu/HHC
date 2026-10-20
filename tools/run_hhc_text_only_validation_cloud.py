"""Wait for all three pilots, then score final40k with zero/requested shapes.

No overlap with MGT pilot training, no new infrastructure, no test selection.
"""
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


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud only')
    config=json.loads(args.config.read_text());control=Path(config['control']);control.mkdir(exist_ok=True)
    if args.detach:
        with (control/'launch_receipt.json').open('x') as receipt,(control/'launch.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            json.dump({'pid':child.pid,'config_sha256':sha(args.config),'started':time.time()},receipt)
        print(json.dumps({'pid':child.pid,'status':'launched_waiting_for_all_mgt'}));return
    with (control/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=control/'STATUS.json'
        if status.exists():raise FileExistsError('Never replay an existing evaluation queue without explicit recovery')
        os.nice(10)
        state={'status':'waiting_for_all_mgt','pid':os.getpid(),'started':time.time(),'phase_exit_codes':{},
               'config_sha256':sha(args.config),'local_checkpoint_downloads':False,'formal_test':False}
        write(status,state)
        env=dict(os.environ,PYTHONPATH='/opt/hhc-vq-deps',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        def check_sources():
            for p,expected in config['sources'].items():assert sha(ROOT/p)==expected,'Source changed: '+p
        def run(phase,command,gpu=False):
            check_sources()
            with (control/f'{phase}.log').open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='0' if gpu else ''),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                state.update(status='running',phase=phase,child_pid=child.pid);write(status,state)
                code=child.wait()
            state['phase_exit_codes'][phase]=code;write(status,state)
            if code:raise RuntimeError(f'{phase} exited {code}; do not automatically replay')
        try:
            while True:
                check_sources();ready=True
                for arm in config['arms']:
                    path=Path(arm['run'])/'training_manifest.json'
                    if not path.exists():ready=False;continue
                    m=json.loads(path.read_text())
                    if m['status']!='pilot_complete_validation_only':ready=False
                    else:assert m['step']==40000 and m['contract']['steps']==40000
                if ready:break
                time.sleep(600)
            while True:
                free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                assert len(free.splitlines())==1
                if int(free.strip())>=13*1024:break
                state['status']='waiting_for_gpu_headroom';write(status,state);time.sleep(600)
            rows=[]
            for arm in config['arms']:
                for policy in ('zero','requested'):
                    name=arm['name']+'_'+policy;out=Path(config['output'])/name
                    out.parent.mkdir(exist_ok=True)
                    run(name+'_generate',[sys.executable,'-u','tools/export_hhc_mgt_validation_cloud.py',
                        '--mgt-run',arm['run'],'--vq-spec',config['vq_spec'],'--output',str(out),
                        '--shape-policy',policy,'--selection','final'],gpu=True)
                    run(name+'_score',[sys.executable,'-u','tools/score_frozen_interhuman_h5_cpu.py',
                        '--generations',str(out/'generations.h5'),'--output',str(out/'metrics.json'),
                        '--reference-embeddings',config['reference_embeddings']])
                    report=json.loads((out/'metrics.json').read_text())
                    rows.append({'arm':arm['name'],'shape_policy':policy,'metrics':report['metrics'],
                        'report_sha256':sha(out/'metrics.json'),'reference_gt_text':report['reference_gt_text']})
                    write(control/'comparison.json',{'status':'partial','rows':rows,'formal_test':False})
            write(control/'comparison.json',{'status':'complete','rows':rows,'formal_test':False,
                'primary_condition':'zero shape; final40000; same seed/request/evaluator as three baselines',
                'interpretation':'val pilot only; requested GT-shape rows are additional-condition diagnostics; test/MM/CI needed for SOTA'})
            state.update(status='complete_val_only',ended=time.time());write(status,state)
        except Exception as error:
            state.update(status='failed',error=repr(error),ended=time.time());write(status,state);raise


if __name__=='__main__':main()
