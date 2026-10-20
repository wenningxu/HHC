"""CPU validation of a full official checkpoint for an explicit host handoff."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/workspace/HHC')
RUN=ROOT/'output/intergen_official_baseline_20261017'

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not args.receipt.exists()
    launch=json.loads((RUN/'LAUNCH.json').read_text())
    provenance=json.loads((ROOT/'output/intergen_official_baseline_setup_20261017/SOURCE_PROVENANCE.json').read_text())
    for name, expected in provenance['tracked_files'].items():
        expected=launch.get('source_changes',{}).get(name,{}).get('after',expected)
        assert sha(Path(launch['cwd'])/name)==expected,name
    assert sha(ROOT/'tools/run_official_intergen_checkpointed_cloud.py')==launch['runtime_changes']['wrapper_sha256']
    original=args.checkpoint
    before=original.stat()
    assert time.time()-before.st_mtime>5
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    snapshot=args.receipt.parent/original.name
    if snapshot!=original:
        assert not snapshot.exists()
        os.link(original,snapshot)
    import torch
    torch.set_num_threads(2)
    saved=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
    step=int(saved['global_step']);epoch=int(saved['epoch'])
    assert 255200<=step<=704000 and step%352==0 and (epoch+1)*352==step
    assert tuple(saved['state_dict']['model.decoder.net.motion_embed.weight'].shape)==(1024,262)
    assert tuple(saved['state_dict']['model.decoder.net.out.linear.weight'].shape)==(262,1024)
    assert all(torch.isfinite(v).all() for v in saved['state_dict'].values() if torch.is_tensor(v))
    optimizer=saved['optimizer_states'][0]
    steps=[int(x['step']) for x in optimizer['state'].values()]
    assert len(steps)==196 and min(steps)==max(steps)==step
    assert all(torch.isfinite(v).all() for item in optimizer['state'].values() for v in item.values() if torch.is_tensor(v))
    scheduler=saved['lr_schedulers'][0]
    assert scheduler['warmup']==10 and scheduler['max_num_iters']==2000 and scheduler['last_epoch']==epoch
    progress=saved['loops']['fit_loop']['epoch_progress']['current']
    assert progress['processed']==epoch+1 and progress['completed']==epoch
    assert before.st_size==original.stat().st_size==snapshot.stat().st_size
    assert not torch.cuda.is_initialized()
    result={'status':'verified_full_resume_candidate','step':step,'epoch':epoch,
            'snapshot':str(snapshot),'original':str(original),'bytes':snapshot.stat().st_size,
            'sha256':sha(snapshot),'optimizer_states':len(steps),'scheduler':scheduler,
            'loops':saved['loops'],'cpu_only':True,'time_unix':time.time(),
            'source_commit':launch['source_commit'],'rng_dataloader_bitwise_claimed':False}
    with args.receipt.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({k:v for k,v in result.items() if k not in ('loops','scheduler')}))

if __name__=='__main__':
    main()
