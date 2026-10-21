"""CPU waiter dispatching the unchanged official eval.py only after final704k."""
import argparse,fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
from pause_hhc_for_intergen_cloud import read,write,sha
from resume_hhc_after_official_intergen_cloud import process,same_live,blocking_jobs,check_pins
from official_intergen_release_contract import parse_official_reproduction_log
ROOT=Path('/workspace/HHC')
WAITING={'waiting_for_official_training','waiting_for_final_checkpoint','waiting_for_gpu_headroom'}

def validate(config,deep=False):
 for path,expected in config['pins'].items():assert sha(path)==expected,path
 hhc=read(config['hhc_queue_config']);assert sha(config['hhc_queue_config'])==config['hhc_queue_config_sha256']
 _,baseline=check_pins(hhc,config['hhc_dispatcher_source_sha256'])
 source=Path(config['source']);provenance=read(config['provenance'])
 assert provenance['source_commit']==hhc['source_commit'] and provenance['no_python_changes'] is True
 assert set(provenance['modified'])=={'configs/model.yaml'}
 for relative,expected in provenance['tracked_files'].items():
  assert sha(source/relative)==provenance['modified'].get(relative,{}).get('after',expected),relative
 assert read(config['setup_status'])['status']=='ready_for_final_training_gated_eval20'
 preflight=read(config['cpu_preflight'])
 assert preflight['status']=='cpu_preflight_passed_no_gpu_execution' and preflight['cpu_cuda_initialized'] is False
 assert preflight['test_dataset_rows']==1098 and preflight['main_samples_per_rep']==1056
 assert preflight['test_ids_sha256']==config['expected_test_ids_sha256']
 assert sha(provenance['evaluator_checkpoint'])==hhc['evaluator_checkpoint_sha256']
 if deep:
  import hashlib
  view=Path(provenance['view'])
  links={str(p.relative_to(view)):str(p.resolve()) for p in view.rglob('*') if p.is_symlink()}
  assert hashlib.sha256(json.dumps(links,sort_keys=True).encode()).hexdigest()==provenance['view_links_sha256']
 return hhc,baseline,provenance,preflight

def final_ready(config,hhc,baseline,control):
 jobs=blocking_jobs(hhc)
 if jobs:return None,{'status':'waiting_for_official_training','blocking_jobs':jobs}
 launch_identity=process(baseline['pid'])
 assert not (launch_identity and launch_identity['birth']==baseline['birth'] and launch_identity['state'] not in ('Z','X'))
 log=Path(baseline['log_path']).read_bytes()[-65536:].decode(errors='replace')
 if not re.search(r'Trainer[.]fit.*max_epochs=2000.*reached',log):
  return None,{'status':'waiting_for_official_training','reason':'training_terminal_without_complete704k_evidence_no_auto_restart'}
 final_path=Path(hhc['baseline_run'])/'FINAL_CHECKPOINT_ACCEPTANCE.json'
 if not final_path.exists():
  # The already-bound HHC CPU waiter owns final verification before checking
  # for the evaluation review. Avoid two verifiers racing to create hardlinks.
  return None,{'status':'waiting_for_final_checkpoint','reason':'waiting_for_existing_HHC_CPU_waiter_final_acceptance_no_duplicate_verifier'}
 final=read(final_path)
 assert final['status']=='verified_final' and final['step']==704000 and final['input_dim']==262
 assert final['source_commit']==hhc['source_commit'] and final['runtime_wrapper_sha256']==hhc['runtime_wrapper_sha256']
 assert final['attributed_trainer_live'] is False and final['cpu_only'] is True and final['cpu_cuda_initialized'] is False
 return final,{'status':'final_official_checkpoint_accepted','step':704000}

def can_recover(previous,control):
 return bool(previous and previous.get('status') in WAITING and not same_live(previous)
             and not previous.get('child_pid') and 'child_exit_code' not in previous
             and not previous.get('dispatch_started') and not (control/'evaluation.launch_intent.json').exists())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--detach',action='store_true');parser.add_argument('--check-only',action='store_true');parser.add_argument('--recover-waiting',action='store_true');args=parser.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and Path(__file__).resolve().parents[1]==ROOT
 config=read(args.config);config_sha=sha(args.config);source_sha=sha(Path(__file__))
 hhc,baseline,provenance,preflight=validate(config,deep=True)
 control=Path(config['control']);assert control.is_relative_to(ROOT/'output');control.mkdir(exist_ok=True)
 status=control/'STATUS.json';intent=control/'evaluation.launch_intent.json'
 if args.check_only:
  final,observed=final_ready(config,hhc,baseline,control)
  print(json.dumps({'status':'cpu_eval_queue_preflight_passed_no_gpu_child','ready':bool(final),'observed':observed,'source':config['source'],'test_rows':preflight['test_dataset_rows'],'main_samples_per_rep':preflight['main_samples_per_rep'],'original_eval_source_sha256':provenance['eval_source_sha256']}),flush=True);return
 if args.detach:
  assert not intent.exists(),'Evaluation dispatch intent exists; never replay'
  if status.exists():assert args.recover_waiting and can_recover(read(status),control)
  name='recovery_'+str(time.time_ns())+'.json' if args.recover_waiting else 'launch_receipt.json'
  with (control/name).open('x') as receipt,(control/'waiter.log').open('a') as log:
   command=[sys.executable,'-u',str(Path(__file__)), '--config',str(args.config.resolve())]
   if args.recover_waiting:command.append('--recover-waiting')
   child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   json.dump({'pid':child.pid,'config_sha256':config_sha,'time_unix':time.time()},receipt)
  print(json.dumps({'pid':child.pid,'status':'cpu_only_official_eval_waiter_launched'}),flush=True);return
 with (control/'waiter.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not intent.exists(),'Evaluation dispatch intent exists; never replay'
  if status.exists():assert args.recover_waiting and can_recover(read(status),control)
  own=process(os.getpid());state={'status':'waiting_for_official_training','pid':os.getpid(),'birth':own['birth'],'started':time.time(),'config_sha256':config_sha,'source_sha256':source_sha,'dispatch_started':False,'local_checkpoint_downloads':False}
  write(status,state)
  def unchanged():
   assert sha(args.config)==config_sha and sha(Path(__file__))==source_sha
   return validate(config)
  try:
   while True:
    hhc,baseline,provenance,preflight=unchanged();final,observed=final_ready(config,hhc,baseline,control)
    state.update(status=observed['status'],last_training_observation=observed);write(status,state)
    if final:break
    time.sleep(600)
   while True:
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines();assert len(free)==1
    if not apps.strip() and int(free[0])>=20*1024:break
    state.update(status='waiting_for_gpu_headroom');write(status,state);time.sleep(600);unchanged();assert final_ready(config,hhc,baseline,control)[0]
   unchanged();validate(config,deep=True);assert sha(final['snapshot'])==final['sha256']
   reproduction=Path(hhc['baseline_run'])/'reproduction';reproduction.mkdir(exist_ok=True)
   checkpoint=Path(provenance['final_checkpoint_link']);assert not checkpoint.exists() and not checkpoint.is_symlink()
   checkpoint.symlink_to(final['snapshot'])
   raw=Path(hhc['reproduction_log']);assert not raw.exists()
   source=Path(config['source']);link=source/'evaluation_1.log';assert not link.exists() and not link.is_symlink();link.symlink_to(raw)
   command=[config['official_python'],'-u','tools/eval.py']
   with intent.open('x') as f:
    json.dump({'command':command,'cwd':str(source),'checkpoint_sha256':final['sha256'],'config_sha256':config_sha,'time_unix':time.time()},f);f.flush();os.fsync(f.fileno())
   directory=os.open(control,os.O_DIRECTORY)
   try:os.fsync(directory)
   finally:os.close(directory)
   state.update(status='evaluation_dispatch_intent_written',dispatch_started=True);write(status,state)
   env={**os.environ,'CUDA_VISIBLE_DEVICES':config.get('cuda_visible_devices','0'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MPLBACKEND':'Agg'}
   with (reproduction/'evaluation.stdout.log').open('x') as log:
    child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    identity=process(child.pid);state.update(status='evaluating_original_official_eval20',child_pid=child.pid,child_birth=identity['birth'] if identity else None);write(status,state)
    code=child.wait()
   state.update(child_exit_code=code);write(status,state);assert code==0,'Official evaluator failed; no automatic rerun or release'
   unchanged();assert sha(checkpoint)==final['sha256']
   parsed=parse_official_reproduction_log(raw.read_text())
   report={'status':'complete','protocol':'official_intergen_eval20','checkpoint_sha256':final['sha256'],'checkpoint_step':final['step'],'raw_log_sha256':sha(raw),'source_commit':hhc['source_commit'],'eval_source_sha256':provenance['eval_source_sha256'],'evaluator_checkpoint_sha256':hhc['evaluator_checkpoint_sha256'],'replication_times':20,'batch_size':96,'main_samples_per_rep':1056,'mm_num_samples':100,'mm_num_repeats':30,'metrics':parsed['metrics'],'per_rep_metrics':parsed['rows'],'test_ids_sha256':preflight['test_ids_sha256'],'metric_precision_note':'Parsed original log rounds each metric to4 decimals; means/CIs recomputed from those rounded values and cross-checked against original summaries. Preserve raw log.','cpu_preflight_sha256':sha(config['cpu_preflight']),'source_provenance_sha256':sha(config['provenance']),'eval_process_exit_code':code,'time_unix':time.time(),'review':'pending independent result review; this dispatcher never creates official_reproduction_acceptance.json','scope':provenance['scope']}
   target=Path(hhc['reproduction_report']);assert not target.exists();write(target,report)
   state.update(status='evaluation_complete_pending_independent_review',report_sha256=sha(target),ended=time.time());write(status,state)
  except BaseException as error:
   state.update(status='failed_needs_inspection_no_automatic_replay',error=repr(error),ended=time.time());write(status,state);raise
if __name__=='__main__':main()
