"""Mock orchestration failure gates; never launch a GPU child."""
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_hhc_train_shape_validation_cloud as driver


def main():
    assert str(ROOT)=='/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    passed=[]
    def rejects(name,action):
        try:action()
        except (AssertionError,RuntimeError,FileExistsError):passed.append(name)
        else:raise AssertionError('Unsafe condition accepted: '+name)
    with tempfile.TemporaryDirectory(prefix='shape_queue_mock_',dir=ROOT/'output') as folder:
        folder=Path(folder);state_path=folder/'upstream.json';proc=folder/'proc'
        state={'status':'waiting_for_all_mgt','pid':42,'config_sha256':'expected'}
        write=lambda:state_path.write_text(json.dumps(state))
        command=proc/'42/cmdline';command.parent.mkdir(parents=True)
        command.write_bytes(b'python\0tools/run_hhc_text_only_validation_cloud.py\0');write()
        assert driver.upstream_ready(state_path,'expected',proc) is False
        passed.append('live_upstream_waits_without_launch')
        command.unlink();rejects('missing_upstream_stops_without_replay',lambda:driver.upstream_ready(state_path,'expected',proc))
        state.update(status='complete_val_only',phase_exit_codes={str(i):0 for i in range(12)});write()
        assert driver.upstream_ready(state_path,'expected',proc) is True
        passed.append('all_12_upstream_phases_required')
        state['phase_exit_codes']['11']=1;write()
        rejects('failed_upstream_phase_rejected',lambda:driver.upstream_ready(state_path,'expected',proc))
        state.update(status='failed');write()
        rejects('explicit_upstream_failure_rejected',lambda:driver.upstream_ready(state_path,'expected',proc))
        rejects('upstream_config_mismatch_rejected',lambda:driver.upstream_ready(state_path,'other',proc))
        assert not driver.gpu_ready('12000\n') and driver.gpu_ready('20000\n')
        rejects('multiple_gpu_inventory_rejected',lambda:driver.gpu_ready('20000\n20000\n'))
        config=json.loads((ROOT/'migration/hhc-train-shape-validation.json').read_text())
        state.update(status='complete_val_only',config_sha256=config['upstream_config_sha256'],
                     phase_exit_codes={str(i):0 for i in range(12)});write()
        zero=folder/'zero.json';zero.write_text(json.dumps({'status':'complete','samples':480,'metrics':{},
            'generation_metadata':{'shape_policy':'zero','mgt_step':40000}}))
        config.update(control=str(folder/'control'),upstream_status=str(state_path),zero_reference=str(zero),output=str(folder/'generated'))
        (folder/'control').mkdir();cp=folder/'config.json';cp.write_text(json.dumps(config));launched=[]
        class FailedChild:
            pid=424242
            def __init__(self,command,**kwargs):launched.append(command)
            def wait(self):return 9
        with patch.object(sys,'argv',['queue','--config',str(cp)]),patch.object(driver.subprocess,'check_output',return_value='20000\n'),patch.object(driver.subprocess,'Popen',FailedChild):
            rejects('nonzero_child_stops_remaining_policies',driver.main)
            result=json.loads((folder/'control/STATUS.json').read_text())
            assert result['status']=='failed' and result['phase_exit_codes']=={'train_mean_generate':9} and len(launched)==1
            rejects('existing_queue_not_replayed',driver.main)
            assert len(launched)==1
    result={'status':'passed_cpu_orchestration_mock','checks':passed,'gpu_work':False,'real_child_launches':0,
            'is_quality_evaluation':False,'queue_source_sha256':driver.sha(Path(driver.__file__)),
            'config_sha256':driver.sha(ROOT/'migration/hhc-train-shape-validation.json'),
            'audit_source_sha256':driver.sha(Path(__file__))}
    destination=ROOT/'output/hhc_sota_20261016/train_shape_queue_cpu_audit.json'
    assert not destination.exists();destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
