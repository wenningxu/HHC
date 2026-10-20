"""After existing zero/requested scoring, run A-only train-prior val ablations."""
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def upstream_ready(path,expected_config_sha,proc_root=Path('/proc')):
    state=json.loads(Path(path).read_text())
    assert state['config_sha256']==expected_config_sha,'Upstream config identity changed'
    if state['status']=='complete_val_only':
        codes=state['phase_exit_codes']
        assert len(codes)==12 and all(code==0 for code in codes.values()),'Upstream incomplete exits'
        return True
    if state['status'] not in ('waiting_for_all_mgt','waiting_for_gpu_headroom','running'):
        raise RuntimeError('Upstream not active/complete: '+state['status'])
    command=proc_root/str(state['pid'])/'cmdline'
    if not command.exists() or b'run_hhc_text_only_validation_cloud.py' not in command.read_bytes():
        raise RuntimeError('Upstream driver missing; inspect its original state, never replay it')
    return False


def gpu_ready(output):
    lines=output.splitlines();assert len(lines)==1,'Single existing GPU required'
    return int(lines[0].strip())>=13*1024


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
        print(json.dumps({'pid':child.pid,'status':'launched_waiting_for_zero_shape_scoring'}));return
    with (control/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=control/'STATUS.json'
        if status.exists():raise FileExistsError('Existing queue needs explicit recovery; never replay')
        os.nice(10)
        state={'status':'waiting_for_zero_shape_scoring','pid':os.getpid(),'started':time.time(),
               'phase_exit_codes':{},'config_sha256':sha(args.config),'local_checkpoint_downloads':False,'formal_test':False}
        write(status,state)
        def check_sources():
            for path,digest in config['sources'].items():assert sha(ROOT/path)==digest,'Source changed: '+path
            assert sha(config['shape_prior'])==config['shape_prior_spec_sha256']
            prior=json.loads(Path(config['shape_prior']).read_text())
            assert sha(prior['array'])==prior['array_sha256']==config['shape_array_sha256']
        env=dict(os.environ,PYTHONPATH='/opt/hhc-vq-deps',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        def run(phase,command,gpu=False):
            check_sources()
            with (control/f'{phase}.log').open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='0' if gpu else ''),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                state.update(status='running',phase=phase,child_pid=child.pid);write(status,state)
                code=child.wait()
            state['phase_exit_codes'][phase]=code;write(status,state)
            if code:raise RuntimeError(f'{phase} exited {code}; no automatic replay')
        try:
            while True:
                check_sources()
                if upstream_ready(config['upstream_status'],config['upstream_config_sha256']):break
                time.sleep(600)
            zero=json.loads(Path(config['zero_reference']).read_text())
            assert zero['status']=='complete' and zero['samples']==480
            assert zero['generation_metadata']['shape_policy']=='zero' and zero['generation_metadata']['mgt_step']==40000
            while not gpu_ready(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)):
                state['status']='waiting_for_gpu_headroom';write(status,state);time.sleep(600)
            rows=[{'shape_policy':'zero','metrics':zero['metrics'],'report_sha256':sha(config['zero_reference'])}]
            for policy in ('train_mean','empirical_train_pair'):
                out=Path(config['output'])/policy;out.parent.mkdir(exist_ok=True)
                run(policy+'_generate',[sys.executable,'-u','tools/export_hhc_mgt_train_prior_validation_cloud.py',
                    '--mgt-run',config['mgt_run'],'--vq-spec',config['vq_spec'],'--shape-prior',config['shape_prior'],
                    '--output',str(out),'--shape-policy',policy,'--selection','final'],gpu=True)
                run(policy+'_score',[sys.executable,'-u','tools/score_frozen_interhuman_h5_cpu.py',
                    '--generations',str(out/'generations.h5'),'--output',str(out/'metrics.json'),
                    '--reference-embeddings',config['reference_embeddings']])
                report=json.loads((out/'metrics.json').read_text())
                for key in ('mgt_checkpoint_sha256','vq_checkpoint_sha256'):
                    assert report['generation_metadata'][key]==zero['generation_metadata'][key]
                assert all(report['reference_gt_text'][k]['equal'] for k in ('GT','text'))
                rows.append({'shape_policy':policy,'metrics':report['metrics'],'report_sha256':sha(out/'metrics.json')})
                write(control/'comparison.json',{'status':'partial','rows':rows,'formal_test':False})
            write(control/'comparison.json',{'status':'complete','arm':'A_historical','rows':rows,'formal_test':False,
                'scope':'Same final40000 MGT/VQ; train-only shape inference factors. Val screening only; no test/MM or SOTA claim.'})
            state.update(status='complete_val_only',ended=time.time());write(status,state)
        except Exception as error:
            state.update(status='failed',error=repr(error),ended=time.time());write(status,state);raise


if __name__=='__main__':main()
