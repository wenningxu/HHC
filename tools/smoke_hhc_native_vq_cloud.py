"""Four native-HHC VQ updates on cloud only, without saving a checkpoint.

Uses the original batch256/window64/fp32 objective. This is a readiness test,
not a reconstruction benchmark or a replacement for the unavailable VQ185k.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dataset.interhuman_interx_raw import InterHumanInterXRaw
from models.vqvae_interhuman_smpl_raw import NativeSMPLFK,VQSMPLRaw,RawLoss,RawNormalizer


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    pending=path.with_suffix('.tmp');pending.write_text(json.dumps(value,indent=2)+'\n');os.replace(pending,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud GPU execution only')
    args.output.mkdir(exist_ok=False)
    status=args.output/'STATUS.json';started=time.time()
    state={'status':'preparing_cpu_data','pid':os.getpid(),'started':started,'seed':123,
           'checkpoint_saved':False,'batch':256,'window':64,'updates':4,'is_quality_evaluation':False}
    write(status,state)
    try:
        torch.set_num_threads(2);torch.manual_seed(123);np.random.seed(123);random.seed(123)
        data_path=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
        fk=NativeSMPLFK();data=InterHumanInterXRaw(data_path,'train',fk)
        loader=torch.utils.data.DataLoader(data,batch_size=256,shuffle=True,num_workers=0,drop_last=True)
        iterator=iter(loader)
        state.update(status='waiting_for_gpu_validation_lock',train_sequences=len(data.ids),windows=len(data));write(status,state)
        with (ROOT/'output/interhuman_baseline_restore_20261016/gpu_validation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert torch.cuda.is_available()
            free,_=torch.cuda.mem_get_info();assert free>=8*1024**3
            # Protect the pre-existing training process. An over-budget smoke
            # fails itself instead of consuming the entire 24GiB device.
            torch.cuda.set_per_process_memory_fraction(.5,0)
            net=VQSMPLRaw(SimpleNamespace(nb_code=12288,mu=.99)).cuda()
            fk=fk.cuda();loss_fn=RawLoss(SimpleNamespace(motion_rep='raw'),fk,RawNormalizer(data.mean,data.std,'cuda:0'))
            optimizer=torch.optim.AdamW(net.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=2e-5)
            state.update(status='gpu_smoke_running',parameters=sum(p.numel() for p in net.parameters()));write(status,state)
            rows=[]
            for update in range(1,5):
                for group in optimizer.param_groups:group['lr']=.0002*(update+1)/1001
                x,beta,gender=(v.cuda() for v in next(iterator))
                prediction,commit,perplexity=net(x,beta,gender)
                components=loss_fn(prediction,x,beta)
                loss=sum(components.values())+.02*commit
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
                optimizer.step()
                row={'update':update,'loss':float(loss),'commit':float(commit),
                     'perplexity':perplexity.detach().cpu().tolist(),
                     'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                     'components':{k:float(v) for k,v in components.items()}}
                rows.append(row);print(json.dumps(row),flush=True)
            state.update(status='smoke_complete',rows=rows,ended=time.time(),
                sources={p:sha(ROOT/p) for p in ('tools/smoke_hhc_native_vq_cloud.py','models/vqvae_interhuman_smpl_raw.py','dataset/interhuman_interx_raw.py','utils/losses.py')},
                dataset_manifest_sha256=sha(data_path/'manifest.json'))
            np.savez(args.output/'normalization.npz',mean=data.mean.numpy(),std=data.std.numpy())
            write(status,state)
    except Exception as error:
        state.update(status='failed',error=repr(error),ended=time.time());write(status,state);raise


if __name__=='__main__':main()
