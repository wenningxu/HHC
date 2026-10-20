"""Rebind only never-dispatched CPU queues after an accepted cross-host resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from restore_official_waiters_after_restart_cloud import sha, read, write_new, proc

ROOT=Path('/workspace/HHC')
RUN=ROOT/'output/intergen_official_baseline_20261017'
MIG=RUN/'edmond_handoff_20261020'

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    launch=read(RUN/'LAUNCH.json')
    trainer=proc(launch['pid'])
    assert trainer and trainer['birth']==launch['birth'] and trainer['state'] not in ('T','Z','X')
    proof=read(MIG/'training_events/full_state_restored.json')
    assert proof['optimizer_states']==196
    assert read(MIG/'training_events/first_resumed_batch.json')['global_step']==proof['global_step']
    write_new(MIG/'waiter_rebind_intent.json',{'time':time.time(),'trainer':trainer})
    hpath=ROOT/'migration/hhc-after-official-intergen-resume.json'
    epath=ROOT/'migration/official-intergen-eval20-after-final.json'
    hhc,evaluation=read(hpath),read(epath)
    write_new(MIG/'hhc_config.runpod.json',hhc)
    write_new(MIG/'eval_config.runpod.json',evaluation)
    hhc['pins'][str(RUN/'LAUNCH.json')]=sha(RUN/'LAUNCH.json')
    for path in [ROOT/'tools/resume_official_intergen_on_edmond.py',
                 MIG/'destination_checkpoint_acceptance.json',MIG/'destination_preflight.json']:
        hhc['pins'][str(path)]=sha(path)
    hpath.write_text(json.dumps(hhc,indent=2)+'\n')
    evaluation['hhc_queue_config_sha256']=sha(hpath)
    epath.write_text(json.dumps(evaluation,indent=2)+'\n')
    records={}
    for name,source,config in [
        ('hhc','resume_hhc_after_official_intergen_cloud.py',hpath),
        ('eval','run_official_intergen_reproduction_cloud.py',epath)]:
        control=Path(read(config)['control'])
        previous=read(control/'STATUS.json')
        actual=proc(previous['pid'])
        assert not actual or actual['birth']!=previous.get('birth') or actual['state'] in ('Z','X')
        assert previous['status']=='waiting_for_official_training'
        assert not previous.get('phase') and not previous.get('child_pid') and not previous.get('phase_exit_codes')
        assert not list(control.glob('*intent*'))
        write_new(MIG/(name+'_status.runpod.json'),previous)
        command=[sys.executable,'-u',str(ROOT/'tools'/source),'--config',str(config),'--recover-waiting','--detach']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        write_new(MIG/(name+'_queue_execution.json'),{'code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        assert result.returncode==0,result.stderr
        launched=json.loads(result.stdout.strip().splitlines()[-1])
        records[name]=proc(launched['pid'])
    for name,source,waiting in [
        ('hhc-text-only-validation','run_hhc_text_only_validation_cloud.py','waiting_for_all_mgt'),
        ('hhc-train-shape-validation-v2','run_hhc_train_shape_validation_v2_cloud.py','waiting_for_zero_shape_scoring')]:
        config=ROOT/'migration'/(name+'.json')
        c=read(config);control=Path(c['control']);status=control/'STATUS.json';previous=read(status)
        assert previous['status']==waiting and previous['config_sha256']==sha(config)
        assert not previous.get('phase') and not previous.get('child_pid') and not previous.get('phase_exit_codes')
        assert not list(control.glob('*intent*'))
        for path,expected in c['sources'].items():assert sha(ROOT/path)==expected,path
        actual=proc(previous['pid'])
        assert not actual or actual['state'] in ('Z','X') or source not in actual['command']
        backup=control/'STATUS.before_edmond_20261020.json'
        assert not backup.exists()
        status.rename(backup)
        with (control/'launch.edmond_20261020.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools'/source),'--config',str(config)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records[name]=proc(child.pid)
    with (MIG/'observer.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/observe_official_intergen_baseline_cloud.py'),'--watch'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    records['observer']=proc(child.pid)
    watcher={'pid':child.pid,'birth':records['observer']['birth'],'time_unix':time.time(),
             'source':str(ROOT/'tools/observe_official_intergen_baseline_cloud.py'),
             'source_sha256':sha(ROOT/'tools/observe_official_intergen_baseline_cloud.py'),
             'log_path':str(MIG/'observer.log'),'execution_host':'edmond-Z790-Taichi'}
    (RUN/'OBSERVER_LAUNCH.json').write_text(json.dumps(watcher,indent=2)+'\n')
    receipt={'status':'queues_launched_pending_independent_readback','identities':records,
             'launch_sha256':sha(RUN/'LAUNCH.json'),'hhc_config_sha256':sha(hpath),
             'eval_config_sha256':sha(epath),'hhc_saved_step':4000}
    write_new(MIG/'waiter_rebind_receipt.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
