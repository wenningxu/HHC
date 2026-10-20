"""One-shot launch after an independently validated cloud-to-cloud checkpoint copy."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path('/workspace/HHC')
RUN=ROOT/'output/intergen_official_baseline_20261017'
MIG=RUN/'edmond_handoff_20261020'

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def proc(pid):
    p=Path('/proc')/str(pid)
    f=(p/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'birth':f[19],'state':f[0]}

def main():
    receipt=MIG/'destination_checkpoint_acceptance.json'
    r=json.loads(receipt.read_text())
    source=json.loads((MIG/'checkpoint_acceptance.json').read_text())
    assert r['sha256']==source['sha256'] and r['step']==source['step']
    assert json.loads((MIG/'freeze_receipt.json').read_text())['status']=='source_trainer_suspended_pending_destination_acceptance'
    preflight=json.loads((MIG/'destination_preflight.json').read_text())
    assert preflight['dataset_rows']==11274 and preflight['updates_per_epoch']==352
    assert preflight['source_files_verified']>=29 and preflight['cuda_initialized'] is False
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert not apps.strip()
    previous=json.loads((RUN/'LAUNCH.json').read_text())
    (MIG/'LAUNCH.runpod.json').write_text(json.dumps(previous,indent=2)+'\n')
    command=['/workspace/intergen-official-env-py38/bin/python','-u',
             str(ROOT/'tools/resume_official_intergen_on_edmond.py'),
             '--runtime-wrapper',str(ROOT/'tools/run_official_intergen_checkpointed_cloud.py'),
             '--checkpoint-receipt',str(receipt),'--event-dir',str(MIG/'training_events')]
    with (MIG/'launch_intent.json').open('x') as f:
        json.dump({'command':command,'time':time.time(),'checkpoint_sha256':r['sha256']},f)
        f.flush();os.fsync(f.fileno())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    log=RUN/'train.edmond_20261020.log'
    started=time.time()
    with log.open('xb') as f:
        child=subprocess.Popen(command,cwd=previous['cwd'],env=env,
                               stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    identity=proc(child.pid)
    launch=dict(previous,pid=child.pid,birth=identity['birth'],command=command,
                log_path=str(log),time=started,execution_host='edmond-Z790-Taichi',
                status='edmond_full_resume_launched_pending_acceptance',
                previous_launch=str(MIG/'LAUNCH.runpod.json'))
    launch['runtime_changes']=dict(previous['runtime_changes'])
    launch['runtime_changes']['full_state_resume']={
        'wrapper':str(ROOT/'tools/resume_official_intergen_on_edmond.py'),
        'wrapper_sha256':sha(ROOT/'tools/resume_official_intergen_on_edmond.py'),
        'resume_step':r['step'],'checkpoint_receipt':str(receipt),
        'checkpoint_receipt_sha256':sha(receipt),'checkpoint_sha256':r['sha256'],
        'pending_official_epoch_scheduler_hook_replayed_once':True,
        'rng_dataloader_bitwise_equivalence_claimed':False,'event_dir':str(MIG/'training_events')}
    temp=RUN/'LAUNCH.edmond.tmp';temp.write_text(json.dumps(launch,indent=2)+'\n');os.replace(temp,RUN/'LAUNCH.json')
    print(json.dumps({'trainer':identity,'step':r['step'],'launch_sha256':sha(RUN/'LAUNCH.json')}),flush=True)

if __name__=='__main__':
    main()
