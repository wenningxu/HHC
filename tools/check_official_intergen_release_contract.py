"""CPU-only positive and negative tests for the official-reproduction release gate."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from official_intergen_release_contract import parse_official_reproduction_log,validate_release_evidence,waiting_recovery_allowed,WAITING


def log_for(count=20):
 text=''
 for i in range(count):
  text+=f'==================== Replication {i} ====================\n'
  text+=f'---> [InterGen] MM Distance: 3.7000\n---> [InterGen] R_precision: (top 1): 0.3000 (top 2): 0.5000 (top 3): 0.6000 \n---> [InterGen] FID: {5.9+i*.01:.4f}\n---> [InterGen] Diversity: 7.5000\n---> [InterGen] Multimodality: 1.2000\n'
 ci=1.96*.01*math.sqrt((20**2-1)/12)/math.sqrt(20)
 for name,mean,interval in [('MM Distance',3.7,0),('FID',5.995,ci),('Diversity',7.5,0),('MultiModality',1.2,0)]:
  text+=f'========== {name} Summary ==========\n---> [InterGen] Mean: {mean:.4f} CInterval: {interval:.4f}\n'
 text+='========== R_precision Summary ==========\n---> [InterGenger](ignored)\n---> [InterGen](top 1) Mean: 0.3000 CInt: 0.0000;(top 2) Mean: 0.5000 CInt: 0.0000;(top 3) Mean: 0.6000 CInt: 0.0000;\n'
 return text


def main():
 text=log_for();parsed=parse_official_reproduction_log(text)
 assert abs(parsed['metrics']['FID']['mean']-5.995)<1e-12
 expected_ci=1.96*.01*math.sqrt((20**2-1)/12)/math.sqrt(20)
 assert abs(parsed['metrics']['FID']['ci95']-expected_ci)<1e-12
 config={'source_commit':'official-commit','runtime_wrapper_sha256':'wrapper-sha','eval_source_sha256':'eval-sha','evaluator_checkpoint_sha256':'interclip-sha'}
 final={'status':'verified_final','step':704000,'input_dim':262,'source_commit':'official-commit','runtime_wrapper_sha256':'wrapper-sha','attributed_trainer_live':False,'cpu_only':True,'cpu_cuda_initialized':False,'sha256':'final-sha'}
 review={'status':'official_intergen_reproduction_accepted','checkpoint_sha256':'final-sha','report_sha256':'report-sha'}
 report={'status':'complete','protocol':'official_intergen_eval20','checkpoint_sha256':'final-sha','raw_log_sha256':'log-sha','source_commit':'official-commit','replication_times':20,'batch_size':96,'main_samples_per_rep':1056,'mm_num_samples':100,'mm_num_repeats':30,'eval_source_sha256':'eval-sha','evaluator_checkpoint_sha256':'interclip-sha','metrics':parsed['metrics']}
 def check(a,b,c,log):return validate_release_evidence(a,b,c,log,'report-sha','log-sha',config)
 check(final,review,report,text)
 rejected=[]
 cases=[('final',key,value) for key,value in [('status','verified_partial'),('step',703648),('input_dim',268),('source_commit','other'),('runtime_wrapper_sha256','other'),('attributed_trainer_live',True),('cpu_only',False),('cpu_cuda_initialized',True)]]
 cases+=[('review',key,value) for key,value in [('status','pending'),('checkpoint_sha256','other'),('report_sha256','changed')]]
 cases+=[('report',key,value) for key,value in [('status','running'),('protocol','different'),('checkpoint_sha256','other'),('raw_log_sha256','changed'),('source_commit','other'),('replication_times',19),('batch_size',32),('main_samples_per_rep',480),('mm_num_samples',0),('mm_num_repeats',10),('eval_source_sha256','changed'),('evaluator_checkpoint_sha256','changed')]]
 for item,key,value in cases:
  a,b,c=deepcopy(final),deepcopy(review),deepcopy(report);{'final':a,'review':b,'report':c}[item][key]=value
  try:check(a,b,c,text)
  except ValueError:rejected.append(item+'.'+key)
  else:raise AssertionError('Gate accepted '+item+'.'+key)
 logs=[('19_replications',log_for(19)),('missing_mm',text.replace('---> [InterGen] Multimodality: 1.2000\n','',1)),('duplicate_fid',text.replace('---> [InterGen] FID: 5.9000','---> [InterGen] FID: 5.9000\n---> [InterGen] FID: 5.9000',1)),('nan_fid',text.replace('FID: 5.9000','FID: nan',1)),('retrieval_out_of_range',text.replace('(top 1): 0.3000','(top 1): 1.3000',1)),('retrieval_not_monotonic',text.replace('(top 1): 0.3000','(top 1): 0.7000',1)),('inconsistent_summary',text.replace('Mean: 5.9950','Mean: 8.9950',1)),('missing_summary',text.replace('MultiModality Summary','Other Summary',1))]
 for name,log in logs:
  try:check(final,review,report,log)
  except ValueError:rejected.append(name)
  else:raise AssertionError('Gate accepted '+name)
 changed=deepcopy(report);changed['metrics']['FID']['mean']+=1
 try:check(final,review,changed,text)
 except ValueError:rejected.append('report_does_not_match_raw_log')
 else:raise AssertionError('Gate accepted incorrect report metric')
 recovery=[]
 for status in WAITING:
  previous={'status':status,'phase_exit_codes':{}}
  assert waiting_recovery_allowed(previous,False,False)
  for name,prior,live,intents in [('live_waiter',previous,True,False),('intent_with_stale_waiting_status',previous,False,True),('phase_started',{**previous,'phase':'A'},False,False),('phase_completed',{**previous,'phase_exit_codes':{'A':0}},False,False)]:
   assert not waiting_recovery_allowed(prior,live,intents);recovery.append(status+':'+name)
 for name,prior,intents in [('missing_status_with_intent',None,True),('missing_status',None,False),('failed_status',{'status':'failed_needs_explicit_phase_recovery'},False)]:
  assert not waiting_recovery_allowed(prior,False,intents);recovery.append(name)
 result={'schema':'official_intergen_release_contract_cpu_test_v2','status':'passed','positive_cases':1,'rejected_cases':len(rejected),'negative_cases':rejected,'waiting_recovery_positive_cases':len(WAITING),'waiting_recovery_rejections':recovery,'no_models_or_gpu_imported':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'contract_sha256':hashlib.sha256((Path(__file__).parent/'official_intergen_release_contract.py').read_bytes()).hexdigest(),'test_data':'synthetic original-format logs, never production metrics'}
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
