"""Mocked queue safety checks only: no checkpoints, models, or CUDA."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import tools.run_hhc_text_only_validation_cloud as queue


def case(name,*,ready=True,free=20000,source_changed=False,existing=False,code=0):
    calls=[]
    with tempfile.TemporaryDirectory(prefix='hhc_text_queue_cpu_',dir=ROOT/'output') as folder:
        base=Path(folder);control=base/'control';control.mkdir();arms=[]
        for label in ('A','B','C'):
            directory=base/label;directory.mkdir();arms.append({'name':label,'run':str(directory)})
            (directory/'training_manifest.json').write_text(json.dumps({'status':'pilot_complete_validation_only' if ready else 'training',
                'step':40000,'contract':{'steps':40000}}))
        source='tools/run_hhc_text_only_validation_cloud.py'
        config={'control':str(control),'output':str(base/'exports'),'arms':arms,'vq_spec':'not-read-in-mock',
                'reference_embeddings':'not-read-in-mock','sources':{source:'changed' if source_changed else hashlib.sha256((ROOT/source).read_bytes()).hexdigest()}}
        path=base/'config.json';path.write_text(json.dumps(config))
        if existing:(control/'STATUS.json').write_text('{}')
        class Child:
            pid=123456789
            def __init__(self,command,**kwargs):self.command=command;calls.append(command)
            def wait(self):
                if code:return code
                out=Path(self.command[self.command.index('--output')+1])
                if '--mgt-run' in self.command:
                    out.mkdir();assert self.command[self.command.index('--selection')+1]=='final'
                else:out.write_text(json.dumps({'metrics':{},'reference_gt_text':{'GT':{'equal':True},'text':{'equal':True}}}))
                return 0
        error=None
        with patch.object(sys,'argv',['queue','--config',str(path)]),patch.object(queue.os,'nice',return_value=0),\
             patch.object(queue.subprocess,'Popen',Child),patch.object(queue.subprocess,'check_output',return_value=str(free)),\
             patch.object(queue.time,'sleep',side_effect=RuntimeError('stop mocked wait')) as sleep:
            try:queue.main()
            except (RuntimeError,FileExistsError,AssertionError) as caught:error=type(caught).__name__
        status=json.loads((control/'STATUS.json').read_text())
        if name=='normal':
            assert error is None and status['status']=='complete_val_only' and len(calls)==12
            summary=json.loads((control/'comparison.json').read_text());assert len(summary['rows'])==6 and not summary['formal_test']
            assert [c[c.index('--shape-policy')+1] for c in calls if '--shape-policy' in c]==['zero','requested']*3
        elif existing:assert error=='FileExistsError' and not calls and status=={}
        else:
            assert error is not None and status['status']=='failed' and len(calls)==(1 if code else 0)
            if not ready or free<13*1024:assert sleep.call_count==1
            if code:assert list(status['phase_exit_codes'].values())==[code]
        return {'case':name,'passed':True,'mocked_children':len(calls),'gpu_work':False}


if __name__=='__main__':
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Run CPU audit in cloud checkout')
    rows=[case('normal'),case('training_wait',ready=False),case('headroom_wait',free=10000),
          case('existing',existing=True),case('source_changed',source_changed=True),case('child_failure',code=9)]
    result={'status':'passed','scope':'Mocked orchestration only; no model/checkpoint/CUDA loaded','cases':rows}
    out=ROOT/'output/hhc_sota_20261016/text_only_queue_cpu_audit.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
