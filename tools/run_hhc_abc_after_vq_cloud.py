"""Wait for the attributed VQ, then run three val-only, single-factor pilots.

Never restarts training, downloads weights, or treats a waiting timeout as failure.
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
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud compute only')
    config=json.loads(args.config.read_text());control=Path(config['control']);control.mkdir(exist_ok=True)
    if args.detach:
        claim=control/'launch_receipt.json'
        with claim.open('x') as receipt, (control/'launch.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            json.dump({'driver_pid':child.pid,'config_sha256':sha(args.config),'started':time.time()},receipt)
        print(json.dumps({'driver_pid':child.pid,'status':'launched_waiting_for_vq'}));return
    status=control/'STATUS.json'
    with (control/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if status.exists():raise FileExistsError('Existing pipeline requires explicit recovery; never replay its arms')
        os.nice(10)
        state={'status':'waiting_for_complete_vq','pid':os.getpid(),'started':time.time(),
               'config_sha256':sha(args.config),'local_checkpoint_downloads':False,'phase_exit_codes':{}}
        write(status,state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/opt/hhc-vq-deps',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        def sources_unchanged():
            for relative,digest in config['sources'].items():
                if sha(ROOT/relative)!=digest:raise RuntimeError('Queued source changed: '+relative)
        def run(phase,command):
            sources_unchanged()
            with (control/f'{phase}.log').open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                state.update(status='running',phase=phase,child_pid=child.pid);write(status,state)
                code=child.wait()
            state['phase_exit_codes'][phase]=code;write(status,state)
            if code:raise RuntimeError(f'{phase} exited {code}; no automatic restart')
        try:
            while True:
                sources_unchanged()
                manifest=json.loads((Path(config['vq_run'])/'training_manifest.json').read_text())
                assert manifest['contract_sha256']==config['vq_contract_sha256']
                if manifest['status']=='complete':
                    assert manifest['last_step']==300000
                    break
                if manifest['status']!='running':raise RuntimeError('VQ not running/complete: '+manifest['status'])
                cmdline=Path(f'/proc/{manifest["pid"]}/cmdline')
                if not cmdline.exists() or b'train_hhc_native_vq_cloud.py' not in cmdline.read_bytes():
                    raise RuntimeError('VQ trainer exited before completion; recovery must use its existing latest')
                time.sleep(600)
            run('freeze_final',[sys.executable,'tools/freeze_hhc_rebuilt_vq_cloud.py','--run',config['vq_run'],
                                '--output',config['frozen_vq']])
            spec=Path(config['frozen_vq'])/'frozen_vq_spec.json'
            for arm in config['arms']:
                free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if len(free.splitlines())!=1 or int(free.strip())<13*1024:
                    raise RuntimeError('Insufficient measured GPU headroom; leave this arm pending')
                run(arm['name'],[sys.executable,'-u','tools/train_hhc_smpl_cfg_ablation.py','--vq-spec',str(spec),
                    '--output',arm['output'],'--conditioning-arm',arm['conditioning'],'--token-window',arm['window'],
                    '--steps','40000','--microbatch','4','--seed','114514','--eval-device','cpu',
                    '--gpu-memory-fraction','.5','--wandb-mode','disabled'])
                result=json.loads((Path(arm['output'])/'training_manifest.json').read_text())
                assert result['status']=='pilot_complete_validation_only'
            state.update(status='complete_validation_only',ended=time.time());write(status,state)
        except Exception as error:
            state.update(status='failed',error=repr(error),ended=time.time());write(status,state);raise


if __name__=='__main__':main()
