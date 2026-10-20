"""Save a verified full-state boundary and fence the superseded custom trainer."""
import hashlib,json,os,signal,time
from pathlib import Path
ROOT=Path('/workspace/HHC')
RUN=ROOT/'output/intergen_rotation_scratch_runpod_20261015'
OUT=ROOT/'output/intergen_official_baseline_transition_control_20261017'
def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        text=(p/'stat').read_text();fields=text[text.rfind(')')+2:].split()
        return {'pid':pid,'birth':fields[19],'state':fields[0],'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
    except FileNotFoundError:return None

def write(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');os.replace(t,p)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    OUT.mkdir(exist_ok=False)
    original=proc(20042)
    assert original and original['birth']=='466013189' and 'train_intergen_rotation_fast_cloud.py' in original['command']
    assert original['state'] not in ('Z','X','T','t')
    oldwaiter=proc(20344);assert not oldwaiter or oldwaiter['state'] in ('Z','X')
    first=json.loads((RUN/'RECOVERY.json').read_text());minimum=((first['step']//500)+1)*500
    write('STATUS.json',{'status':'waiting_for_next_complete_full_state','minimum_step':minimum,'trainer':original,'coordinator_pid':os.getpid()})
    print(json.dumps({'waiting_for_checkpoint_step':minimum,'coordinator_pid':os.getpid()}),flush=True)
    deadline=time.monotonic()+900
    stopped=False;terminated=False
    try:
        while True:
            current=proc(20042);assert current and current['birth']==original['birth'] and current['state'] not in ('Z','X')
            assert time.monotonic()<deadline,'No complete boundary within15min; original training preserved'
            rec=json.loads((RUN/'RECOVERY.json').read_text());status=json.loads((RUN/'STATUS.json').read_text())
            if rec['step']>=minimum and status['step']==rec['step']:
                os.kill(20042,signal.SIGSTOP);stopped=True;time.sleep(.1)
                if json.loads((RUN/'RECOVERY.json').read_text())==rec and json.loads((RUN/'STATUS.json').read_text())['step']==rec['step']:break
                os.kill(20042,signal.SIGCONT);stopped=False
            time.sleep(.25)
        for name in ('recovery.pt','latest.pth','best.pth'):
            os.link(RUN/name,OUT/name)
        for name in ('config.json','RECOVERY.json','STATUS.json'):
            (OUT/('previous_'+name)).write_bytes((RUN/name).read_bytes())
        full=OUT/'recovery.pt';assert full.stat().st_size==rec['bytes'] and sha(full)==rec['sha256']
        import torch
        torch.set_num_threads(2)
        saved=torch.load(full,map_location='cpu',mmap=True,weights_only=False)
        latest=torch.load(OUT/'latest.pth',map_location='cpu',mmap=True,weights_only=False)
        assert saved['step']==saved['checkpoint']['step']==latest['step']==rec['step']
        assert saved['checkpoint']['config']['scratch'] and latest['config']['scratch']
        assert {'optimizer','python_rng','numpy_rng','torch_rng','cuda_rng'}<=saved.keys()
        assert saved['optimizer']['state']
        assert all(torch.isfinite(v).all() for v in saved['checkpoint']['trainable_state'].values())
        assert all(torch.isfinite(t).all() for v in saved['optimizer']['state'].values() for t in v.values() if isinstance(t,torch.Tensor))
        assert all(torch.equal(v,latest['trainable_state'][k]) for k,v in saved['checkpoint']['trainable_state'].items())
        assert not torch.cuda.is_initialized()
        receipt={'status':'verified','step':rec['step'],'full_state':{'bytes':full.stat().st_size,'sha256':rec['sha256']},'latest':{'bytes':(OUT/'latest.pth').stat().st_size,'sha256':sha(OUT/'latest.pth')},'optimizer_rng_restored_capable':True,'dataloader_cursor_saved':False,'frozen_clip_not_embedded':True,'snapshots':'persistent cloud-volume hardlinks; not independent off-volume backup','cpu_only_verification':True}
        write('pause_checkpoint_acceptance.json',receipt)
        # Stop only after the immutable full-state snapshot passes acceptance.
        current=proc(20042);assert current and current['birth']==original['birth'] and current['state'] in ('T','t')
        os.kill(20042,signal.SIGTERM);os.kill(20042,signal.SIGCONT);stopped=False
        until=time.monotonic()+30
        while True:
            current=proc(20042)
            if not current or current['state'] in ('Z','X'):break
            assert time.monotonic()<until,'Old trainer did not exit'
            time.sleep(.1)
        terminated=True
        status={'status':'paused_by_user_for_official_baseline','step':rec['step'],'trainer_terminal':proc(20042),'saved_state':receipt,'hhc_saved_step':4000,'hhc_resume_authorized_now':False,'new_gpu_training_started':False,'timestamp_unix':time.time()}
        write('STATUS.json',status)
        p=RUN/'STATUS.json';q=dict(status,best_step=latest.get('best_step'),best_fid=latest.get('best_fid'));p.write_text(json.dumps(q,indent=2)+'\n')
        print(json.dumps(status),flush=True)
    finally:
        if stopped and not terminated:
            current=proc(20042)
            if current and current['birth']==original['birth']:os.kill(20042,signal.SIGCONT)
if __name__=='__main__':main()
