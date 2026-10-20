"""CPU-only mocked orchestration checks; never initializes a model or CUDA."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import tools.run_hhc_abc_after_vq_cloud as queue


def case(name,*,exit_code=0,free=20000,changed=False,existing=False,vq_status='complete'):
    calls=[]
    with tempfile.TemporaryDirectory(prefix='hhc_queue_cpu_',dir=ROOT/'output') as folder:
        base=Path(folder);vq=base/'vq';vq.mkdir();control=base/'control';control.mkdir()
        manifest={'status':vq_status,'last_step':300000,'pid':999999999,'contract_sha256':'test'}
        (vq/'training_manifest.json').write_text(json.dumps(manifest))
        source='tools/run_hhc_abc_after_vq_cloud.py'
        config={'control':str(control),'vq_run':str(vq),'vq_contract_sha256':'test','frozen_vq':str(base/'frozen'),
                'sources':{source:'changed' if changed else hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},
                'arms':[{'name':a,'conditioning':c,'window':w,'output':str(base/a)} for a,c,w in
                        [('A','historical','historical'),('B','matched','historical'),('C','historical','uniform')]]}
        path=base/'config.json';path.write_text(json.dumps(config))
        if existing:(control/'STATUS.json').write_text('{}')
        class Child:
            pid=123456789
            def __init__(self,command,**kwargs):self.command=command;calls.append(command)
            def wait(self):
                if exit_code:return exit_code
                if '--conditioning-arm' in self.command:
                    out=Path(self.command[self.command.index('--output')+1]);out.mkdir()
                    (out/'training_manifest.json').write_text(json.dumps({'status':'pilot_complete_validation_only'}))
                return 0
        error=None
        with patch.object(sys,'argv',['queue','--config',str(path)]),patch.object(queue.os,'nice',return_value=0),\
             patch.object(queue.subprocess,'Popen',Child),patch.object(queue.subprocess,'check_output',return_value=str(free)):
            try:queue.main()
            except (RuntimeError,FileExistsError) as caught:error=type(caught).__name__
        status=json.loads((control/'STATUS.json').read_text())
        if name=='normal':
            assert error is None and status['status']=='complete_validation_only' and len(calls)==4
            for call,arm in zip(calls[1:],config['arms']):
                assert call[call.index('--conditioning-arm')+1]==arm['conditioning']
                assert call[call.index('--token-window')+1]==arm['window']
                assert '--formal-test' not in call and call[call.index('--steps')+1]=='40000'
        elif existing:assert error=='FileExistsError' and not calls and status=={}
        else:
            assert error=='RuntimeError' and status['status']=='failed'
            assert len(calls)==(1 if exit_code or free<13*1024 else 0)
        return {'case':name,'passed':True,'mocked_children':len(calls),'gpu_work':False}


if __name__=='__main__':
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Run this CPU-only audit in the cloud checkout')
    rows=[case('normal'),case('existing',existing=True),case('source_changed',changed=True),
          case('freeze_exit_failure',exit_code=7),case('headroom',free=10000),case('vq_exited',vq_status='running')]
    result={'status':'passed','scope':'Mocked queue safety only; no model, checkpoint or CUDA loaded','cases':rows}
    out=ROOT/'output/hhc_sota_20261016/abc_queue_cpu_audit.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
