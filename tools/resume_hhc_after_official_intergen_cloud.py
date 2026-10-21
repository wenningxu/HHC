"""CPU-only saved-A/B/C queue, gated by final official training and reviewed eval20."""
import argparse,fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
from pause_hhc_for_intergen_cloud import read,write,sha,sources
from resume_hhc_after_intergen_fast_v3_cloud import checkpoint_acceptance,arm_command
from official_intergen_release_contract import validate_release_evidence,waiting_recovery_allowed
ROOT=Path(__file__).resolve().parents[1]
GPU_SCRIPTS={'train_intergen_rotation.py','train_intergen_rotation_fast_cloud.py','run_official_intergen_checkpointed_cloud.py','train_hhc_smpl_cfg_ablation.py','diagnose_custom_intergen_heads_cloud.py'}
OLD_DRIVERS={'resume_hhc_after_intergen_cloud.py','resume_hhc_after_intergen_fast_cloud.py','resume_hhc_after_intergen_fast_v3_cloud.py','run_hhc_abc_after_vq_cloud.py'}


def process(pid):
    p=Path('/proc')/str(pid)
    try:
        values=(p/'stat').read_text().rsplit(')',1)[1].split()
        try:cwd=str((p/'cwd').resolve())
        except PermissionError:cwd=None  # Unrelated container service can deny cwd inspection.
        return {'pid':pid,'birth':values[19],'state':values[0],'argv':[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x],'cwd':cwd}
    except FileNotFoundError:return None


def same_live(identity):
    actual=process(identity['pid'])
    return bool(actual and actual['birth']==identity['birth'] and actual['state'] not in ('Z','X'))


def blocking_jobs(config):
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            item=process(int(path.parent.name))
            if not item or item['state'] in ('Z','X'):continue
            names={Path(x).name for x in item['argv']}
            if (names&(GPU_SCRIPTS|OLD_DRIVERS) or
                    ('train.py' in names and item['cwd']==config['baseline_source']) or
                    ('eval.py' in names and 'intergen' in (item['cwd'] or '').lower())):
                found.append(item)
        except (FileNotFoundError,ProcessLookupError):continue
    return found


def check_pins(config,source_sha):
    assert sha(Path(__file__))==source_sha
    for path,expected in config['pins'].items():assert sha(path)==expected,path
    policy=read(config['priority_config'])
    assert policy['policy']=='official_intergen_baseline_exclusive_custom_paused'
    legacy=read(config['legacy_hhc_policy']);sources(legacy)
    assert sha(legacy['vq_spec'])==legacy['vq_spec_sha256']
    baseline=read(Path(config['baseline_run'])/'LAUNCH.json')
    assert baseline['source_commit']==config['source_commit'] and baseline['cwd']==config['baseline_source']
    assert baseline['learning_protocol']['input_dim']==262 and baseline['learning_protocol']['precision']==32
    assert baseline['runtime_changes']['wrapper_sha256']==config['runtime_wrapper_sha256']
    # Keep every original official source byte, allowing only the launch-bound train config.
    catalogue=read(config['official_source_catalogue'])
    for relative,expected in catalogue['tracked_files'].items():
        expected=baseline.get('source_changes',{}).get(relative,{}).get('after',expected)
        assert sha(Path(config['baseline_source'])/relative)==expected,relative
    old=read(config['superseded_waiter_status'])
    assert old['status']=='superseded_by_official_baseline_priority' and not old.get('phase') and not old.get('phase_exit_codes')
    custom=Path(config['custom_run'])
    assert read(custom/'STATUS.json')['status']=='paused_by_user_for_official_baseline'
    assert not (custom/'RECOVERY.json').exists(),'Obsolete automatic resume fence changed'
    return legacy,baseline


def saved_hhc_snapshot(config,legacy,control):
    pause=read(Path(legacy['pause_control'])/'STATUS.json')
    assert pause['status']=='paused_checkpoint_verified_waiting_intergen'
    accepted=pause['saved_checkpoint'];assert accepted['step']==4000 and accepted['contract_sha256']==legacy['hhc_contract_sha256']
    source=Path(accepted['checkpoint']);snapshot=control/'saved_A_step4000.pth'
    if not snapshot.exists():
        assert source.stat().st_size==accepted['bytes'];os.link(source,snapshot)
    actual=checkpoint_acceptance(snapshot,4000,legacy['hhc_contract_sha256'])
    assert actual['sha256']==accepted['sha256'] and actual['bytes']==accepted['bytes']
    write(control/'saved_hhc_checkpoint_acceptance.json',actual)
    return accepted


def readiness(config,baseline,control):
    jobs=blocking_jobs(config)
    if jobs:return False,{'status':'waiting_for_official_training','blocking_jobs':jobs}
    log=Path(baseline['log_path'])
    if not log.exists():return False,{'status':'waiting_for_official_training','reason':'training_log_unavailable'}
    with log.open('rb') as stream:
        stream.seek(0,os.SEEK_END);stream.seek(max(0,stream.tell()-65536));tail=stream.read().decode(errors='replace')
    if not re.search(r'Trainer[.]fit.*max_epochs=2000.*reached',tail):
        return False,{'status':'waiting_for_official_training','reason':'trainer_terminal_without_verified_2000epoch_finish'}
    run=Path(config['baseline_run']);final_path=run/'FINAL_CHECKPOINT_ACCEPTANCE.json'
    if not final_path.exists():
        checkpoints=list((run/'official_intergen_scratch_20261017/model').glob('*.ckpt'))
        if not checkpoints or time.time()-max(p.stat().st_mtime for p in checkpoints)<5:
            return False,{'status':'waiting_for_official_final_checkpoint'}
        result=subprocess.run([sys.executable,'-u',config['final_verifier'],'--final'],cwd=ROOT,capture_output=True,text=True)
        write(control/'official_final_verifier_execution.json',{'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        assert result.returncode==0,'Final official checkpoint rejected; no automatic HHC launch'
    final=read(final_path)
    assert final['status']=='verified_final' and final['step']==704000 and final['input_dim']==262
    review_path=Path(config['review_receipt']);report_path=Path(config['reproduction_report']);eval_log=Path(config['reproduction_log'])
    if not all(p.is_file() for p in (review_path,report_path,eval_log)):
        return False,{'status':'waiting_for_official_reproduction_review','reason':'actual eval20 report/log and reviewed acceptance required'}
    parsed=validate_release_evidence(final,read(review_path),read(report_path),eval_log.read_bytes().decode('utf-8'),sha(report_path),sha(eval_log),config)
    return True,{'status':'official_training_and_reproduction_accepted','step':704000,'final':final,'review_sha256':sha(review_path),'metrics':parsed['metrics']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--detach',action='store_true');parser.add_argument('--recover-waiting',action='store_true');args=parser.parse_args()
    assert str(ROOT)=='/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    config=read(args.config);config_sha=sha(args.config);source_sha=sha(Path(__file__))
    legacy,baseline=check_pins(config,source_sha);abc=read(config['original_abc_config'])
    control=Path(config['control']);assert control.is_relative_to(ROOT/'output');control.mkdir(exist_ok=True)
    path=control/'STATUS.json'
    if args.check_only:
        accepted=saved_hhc_snapshot(config,legacy,control)
        ready,observed=readiness(config,baseline,control)
        print(json.dumps({'status':'preflight_passed_no_gpu_child_launch','release_gate':ready,'observed':observed,'saved_hhc_step':accepted['step'],'A_resume_command':arm_command(abc['arms'][0],legacy['vq_spec'],True),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}),flush=True);return
    if args.detach:
        intents=list(control.glob('*.launch_intent.json'))
        assert not intents,'A dispatch intent exists; never automatically replay an arm'
        if path.exists():
            previous=read(path)
            assert args.recover_waiting and waiting_recovery_allowed(previous,same_live(previous),bool(intents))
        name='recovery_launch_'+str(int(time.time()))+'.json' if args.recover_waiting else 'launch_receipt.json'
        with (control/name).open('x') as receipt,(control/'launch.log').open('a') as log:
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve())]
            if args.recover_waiting:command.append('--recover-waiting')
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            json.dump({'pid':child.pid,'config_sha256':config_sha,'time_unix':time.time()},receipt)
        print(json.dumps({'pid':child.pid,'status':'launched_cpu_waiter_no_gpu_child'}),flush=True);return
    with (control/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(path) if path.exists() else None
        intents=list(control.glob('*.launch_intent.json'))
        assert not intents,'A dispatch intent exists; never automatically replay an arm'
        if previous:
            assert args.recover_waiting and waiting_recovery_allowed(previous,same_live(previous),bool(intents)),'Never replay an executed arm'
        identity=process(os.getpid());state={'status':'waiting_for_official_training','pid':os.getpid(),'birth':identity['birth'],'started':time.time(),'phase_exit_codes':{},'config_sha256':config_sha,'dispatcher_source_sha256':source_sha,'local_checkpoint_downloads':False,'saved_hhc_step':4000}
        write(path,state)
        def unchanged():
            assert sha(args.config)==config_sha
            return check_pins(config,source_sha)
        try:
            unchanged();accepted=saved_hhc_snapshot(config,legacy,control)
            while True:
                legacy,baseline=unchanged();ready,observed=readiness(config,baseline,control)
                state.update(status=observed['status'] if not ready else 'release_evidence_verified',last_official_observation=observed)
                write(path,state)
                if ready:break
                time.sleep(600)
            write(control/'official_release_acceptance.json',observed)
            assert sha(observed['final']['snapshot'])==observed['final']['sha256']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':config.get('cuda_visible_devices','0'),'PYTHONPATH':'/opt/hhc-vq-deps','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
            for index,arm in enumerate(abc['arms']):
                legacy,baseline=unchanged();assert readiness(config,baseline,control)[0]
                out=Path(arm['output'])
                if index==0:
                    manifest=read(out/'training_manifest.json')
                    assert manifest['status']=='paused_by_user_priority' and manifest['step']==accepted['step'] and manifest['contract_sha256']==legacy['hhc_contract_sha256']
                    assert sha(accepted['checkpoint'])==accepted['sha256'],'Original A changed before resume'
                else:assert not out.exists(),'Never replay an existing B/C run'
                while True:
                    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
                    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
                    assert len(free)==1
                    if not apps.strip() and int(free[0])>=13*1024:break
                    state.update(status='waiting_for_gpu_headroom');write(path,state);time.sleep(600);unchanged();assert readiness(config,baseline,control)[0]
                with (control/(arm['name']+'.log')).open('x') as log:
                    command=arm_command(arm,legacy['vq_spec'],index==0)
                    # Write once, durably, BEFORE dispatch. Never remove an intent on failure.
                    intent=control/(arm['name']+'.launch_intent.json')
                    with intent.open('x') as receipt:
                        json.dump({'arm':arm['name'],'command':command,'config_sha256':config_sha,'dispatcher_source_sha256':source_sha,'time_unix':time.time()},receipt)
                        receipt.flush();os.fsync(receipt.fileno())
                    directory=os.open(control,os.O_DIRECTORY)
                    try:os.fsync(directory)
                    finally:os.close(directory)
                    state.update(status='phase_launch_intent_written',phase=arm['name'],child_pid=None,child_birth=None);write(path,state)
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    child_identity=process(child.pid)
                    state.update(status='running',child_pid=child.pid,child_birth=child_identity['birth'] if child_identity else None);write(path,state);code=child.wait()
                state['phase_exit_codes'][arm['name']]=code;write(path,state);assert code==0,'Training failed; never automatically replay a phase'
                manifest=read(out/'training_manifest.json')
                assert manifest['status']=='pilot_complete_validation_only' and manifest['step']==40000
                assert manifest['contract']['vq_spec_sha256']==legacy['vq_spec_sha256'] and manifest['contract']['seed']==114514 and manifest['contract']['steps']==40000
            state.update(status='complete_validation_training_only',ended=time.time());write(path,state)
        except BaseException as error:
            state.update(status='failed_needs_explicit_phase_recovery',error=repr(error),ended=time.time());write(path,state);raise
if __name__=='__main__':main()
