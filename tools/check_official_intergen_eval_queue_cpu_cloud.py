"""CPU-only crash/recovery rejection checks, no model imports."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import run_official_intergen_reproduction_cloud as queue

def main():
 rejected=[];positive=0
 with tempfile.TemporaryDirectory() as folder:
  control=Path(folder)
  with patch.object(queue,'same_live',return_value=False):
   for status in queue.WAITING:
    previous={'status':status,'dispatch_started':False}
    assert queue.can_recover(previous,control);positive+=1
    for key,value in [('dispatch_started',True),('child_pid',123),('child_exit_code',0),('child_exit_code',-9),('status','evaluation_complete_pending_independent_review'),('status','failed_needs_inspection_no_automatic_replay')]:
     assert not queue.can_recover({**previous,key:value},control);rejected.append(status+':'+key+'='+str(value))
    intent=control/'evaluation.launch_intent.json';intent.write_text('{}')
    assert not queue.can_recover(previous,control);rejected.append(status+':durable_intent_with_stale_status');intent.unlink()
   assert not queue.can_recover(None,control);rejected.append('missing_previous_status')
  with patch.object(queue,'same_live',return_value=True):
   assert not queue.can_recover({'status':'waiting_for_official_training'},control);rejected.append('original_waiter_live')
  train_log=control/'train.log';train_log.write_text('`Trainer.fit` stopped: `max_epochs=2000` reached.\n')
  hhc={'baseline_run':str(control)};baseline={'pid':123,'birth':'old-birth','log_path':str(train_log)}
  with patch.object(queue,'blocking_jobs',return_value=[]),patch.object(queue,'process',return_value=None),patch.object(queue.subprocess,'run',side_effect=AssertionError('Must never start a second final verifier')):
   final,observed=queue.final_ready({},hhc,baseline,control)
   assert final is None and observed['status']=='waiting_for_final_checkpoint'
   rejected.append('no_final_receipt_never_dispatch_or_duplicate_final_verifier')
 print(json.dumps({'status':'passed','waiting_recovery_positive_cases':positive,'rejected_count':len(rejected),'rejected':rejected,'no_models_or_gpu_imported':True}),flush=True)
if __name__=='__main__':main()
