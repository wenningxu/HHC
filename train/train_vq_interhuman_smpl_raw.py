"""Train the handless Inter-X raw VQ on native InterHuman SMPL + shape."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from dataset.interhuman_interx_raw import InterHumanInterXRaw
from dataset.interhuman_smpl_pair import to_interhuman
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from models.vqvae_interhuman_smpl_raw import NativeSMPLFK,VQSMPLRaw,RawNormalizer,RawLoss
from models.interhuman_evaluator_wrapper import InterHumanEvaluatorWrapper,load_official_metrics_module
from tools.reconstruct_interhuman_raw_body import load_model
from tools.score_fair_interhuman_main import main_metrics_from_embeddings
from utils.quantizer_state import quantizer_training_state
from utils.runtime import evaluating,preserving_rng_state


def write_json(path,data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)


@torch.no_grad()
def validate(net,data,fk,smpl,adapter,evaluator,metrics,device,count=480):
    """Fixed ID/center-window val panel. Training monitor, not formal test."""
    rows=[]
    geometry_sum=0.
    geometry_count=0
    with preserving_rng_state(),evaluating(net):
        for i in range(count):
            raw=data.motions[i]
            length=min(len(raw)//4*4,300)
            start=(len(raw)-length)//2
            target=raw[start:start+length].to(device)
            beta=data.betas[i][None].to(device)
            gender=torch.ones(1,2,dtype=torch.long,device=device)
            mean,std=data.mean.to(device),data.std.to(device)
            pred,_,_=net(((target-mean)/std)[None],beta,gender)
            pred=(pred[0]*std+mean).reshape(length,23,12)
            target=target.reshape(length,23,12)
            pj=fk(pred[None],beta)
            tj=fk(target[None],beta)
            for p,t in zip(pj,tj):
                geometry_sum+=torch.linalg.vector_norm(p-t,dim=-1).sum().item()
                geometry_count+=p.numel()//3
            pred_native=fk.to_archive(pred,beta).cpu()
            target_native=fk.to_archive(target,beta).cpu()
            generated=to_interhuman(pred_native,beta[0].reshape(2,10).cpu(),smpl,adapter,normalize=True)
            reference=to_interhuman(target_native,beta[0].reshape(2,10).cpu(),smpl,adapter,normalize=True)
            text=(data.root/'texts_processed'/f'{data.ids[i]}.txt').read_text().splitlines()[0]
            rows.append((data.ids[i],text,generated,reference))
        text_emb,pred_emb,gt_emb=[],[],[]
        for begin in range(0,count,96):
            batch=rows[begin:begin+96]
            lengths=torch.tensor([len(r[2]) for r in batch])
            p=torch.zeros(len(batch),int(lengths.max()),524)
            g=torch.zeros_like(p)
            for j,row in enumerate(batch):
                p[j,:lengths[j]]=torch.from_numpy(row[2]); g[j,:lengths[j]]=torch.from_numpy(row[3])
            args=([r[0] for r in batch],[r[1] for r in batch])
            t,z=evaluator.get_co_embeddings((*args,p[...,:262],p[...,262:],lengths))
            ref=evaluator.get_motion_embeddings((*args,g[...,:262],g[...,262:],lengths))
            text_emb.append(t.cpu().numpy()); pred_emb.append(z.cpu().numpy()); gt_emb.append(ref.cpu().numpy())
        result=main_metrics_from_embeddings(np.concatenate(text_emb),np.concatenate(pred_emb),np.concatenate(gt_emb),
                                           metrics,batch_size=96,random_seed=20260130)
        result.update(body_mpjpe_mm=1000*geometry_sum/geometry_count,protocol='fixed_val_first480_center_max300_raw_smpl_v1')
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--eval-device',default='cuda:1')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int,default=123)
    args=p.parse_args()
    args.nb_code=12288; args.mu=.99; args.motion_rep='raw'
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(args.output).free<5*1024**3 and not args.smoke:
        raise RuntimeError('Need 5 GiB free for latest/best/175k checkpoint and atomic saving')
    def log(row):
        row={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**row}
        line=json.dumps(row)
        print(line,flush=True)
        with (args.output/'metrics.jsonl').open('a') as stream: stream.write(line+'\n')
    fk=NativeSMPLFK()
    train=InterHumanInterXRaw(args.data,'train',fk)
    val=InterHumanInterXRaw(args.data,'val',fk,window=None,stats=(train.mean,train.std))
    np.savez(args.output/'normalization.npz',mean=train.mean.numpy(),std=train.std.numpy())
    assert len(val.ids)>=480
    write_json(args.output/'validation_panel.json',{'ids':val.ids[:480],'crop':'center, floor(T/4)*4, max300',
                                                 'caption':'first released line','split':'val','seed':20260130})
    loader=DataLoader(train,batch_size=256,shuffle=True,num_workers=0,drop_last=True,pin_memory=True)
    iterator=iter(loader)
    net=VQSMPLRaw(args).to(args.device)
    fk=fk.to(args.device)
    objective=RawLoss(args,fk,RawNormalizer(train.mean,train.std,args.device))
    optimizer=torch.optim.AdamW(net.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=2e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300000,eta_min=0)
    sources=['train/train_vq_interhuman_smpl_raw.py','models/vqvae_interhuman_smpl_raw.py',
             'dataset/interhuman_interx_raw.py','models/encdec.py','models/quantize_cnn.py',
             'models/vqvae_sep3_shape.py','utils/losses.py','dataset/interhuman_smpl_pair.py',
             'models/interhuman_evaluator_wrapper.py','utils/quantizer_state.py']
    manifest={'status':'smoke_running' if args.smoke else 'running','pid':os.getpid(),
              'schema':'interhuman_interx_raw_handless_v1','seed':args.seed,'data':str(args.data),
              'parameters':sum(x.numel() for x in net.parameters()),'batch':256,'window':64,
              'codebook_size':12288,'latent_dims':[96,64],'shape_condition_dim':8,
              'precision':'fp32','lr':.0002,'weight_decay':.00002,'adam_betas':[.9,.99],
              'warmup_updates':999,'main_steps':300000,'eval_interval':5000,
              'loss_weights':{'main':1,'Joint':3,'Velocity':100,'Rotation':.01,'Root':.1,'FootContact':10,'DistanceMap':.1,'commit':.02},
              'sources':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
              'dataset_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
              'notes':['Original Inter-X raw slots, no hand channel; no speed features as model inputs.',
                       'Y-up global basis and pelvis root; invertible to native SMPL with original beta.',
                       'SMPL beta-dependent FK replaces Inter-X group-specific SMPL-X skeleton.',
                       'Only finite empty-contact safeguard differs from historical loss reduction.',
                       'Fixed InterHuman val monitor; not Inter-X evaluator or formal official test.',
                       'Checkpoint retention: latest, best val FID, step175000 only; do not accumulate milestones.']}
    run=None
    if not args.smoke:
        run=wandb.init(project='NewInter',group='vq_interhuman_smpl_raw',name=args.output.name,
                       dir=str(args.output),mode='online',config=manifest)
        manifest['wandb_run_id']=run.id
    write_json(args.output/'training_manifest.json',manifest)
    log({'event':'ready','windows':len(train),'train_sequences':len(train.ids),'val_sequences':len(val.ids),
         'parameters':manifest['parameters']})
    evaluator=InterHumanEvaluatorWrapper.from_official_checkpoint(args.eval_device)
    metrics=load_official_metrics_module()
    adapter=OfficialInterHumanAdapter()
    smpl=load_model()
    best=float('inf'); best_step=None
    started=time.monotonic()
    for update in range(1, (4 if args.smoke else 300999)+1):
        warmup=update<=999
        step=0 if warmup else update-999
        if warmup:
            for group in optimizer.param_groups: group['lr']=.0002*(update+1)/1001
        try: x,beta,gender=next(iterator)
        except StopIteration:
            iterator=iter(loader); x,beta,gender=next(iterator)
        x,beta,gender=x.to(args.device),beta.to(args.device),gender.to(args.device)
        pred,commit,perplexity=net(x,beta,gender)
        components=objective(pred,x,beta)
        loss=sum(components.values())+.02*commit
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss update{update}: {components}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update<=4 or update%50==0:
            if any(not torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None):
                raise FloatingPointError(f'Nonfinite gradient update{update}')
        optimizer.step()
        if not warmup: scheduler.step()
        if update<=4 or update%200==0:
            row={'event':'train','update':update,'step':step,'warmup':warmup,'loss':loss.item(),
                 'lr':optimizer.param_groups[0]['lr'],'perplexity':perplexity.tolist(),
                 'elapsed_s':time.monotonic()-started,'gpu_peak_gib':torch.cuda.max_memory_allocated(args.device)/1024**3,
                 **{k:v.item() for k,v in components.items()},'commit':commit.item()}
            log(row)
            if run: wandb.log({k:v for k,v in row.items() if isinstance(v,(int,float))},step=update)
        should_eval=(args.smoke and update==4) or (not warmup and step%5000==0)
        if should_eval:
            result=validate(net,val,fk,smpl,adapter,evaluator,metrics,args.device,count=480)
            is_best=result['fid']<best
            if is_best: best,best_step=result['fid'],step
            log({'event':'validation','step':step,'best_fid':best,'best_step':best_step,**result})
            write_json(args.output/f'validation_{step:06d}.json',result)
            if run: wandb.log({'val/'+k:v for k,v in result.items() if isinstance(v,(int,float))},step=update)
        else: is_best=False
        if (not args.smoke and (update==999 or (step>0 and step%1000==0))):
            if shutil.disk_usage(args.output).free<2*1024**3:
                raise RuntimeError('Disk below 2 GiB: stop before overwriting last valid checkpoint')
            state={'net':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'quantizer_training_state':quantizer_training_state(net),'update':update,'step':step,
                   'best_fid':best,'best_step':best_step,'mean':train.mean,'std':train.std,
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'manifest':manifest}
            pending=args.output/'latest.tmp'
            torch.save(state,pending); pending.replace(args.output/'latest.pth')
            for wanted,name in ((is_best,'best_fid.pth'),(step==175000,'step175000.pth')):
                if wanted:
                    torch.save({'net':net.state_dict(),'step':step,'mean':train.mean,'std':train.std,'manifest':manifest},pending)
                    pending.replace(args.output/name)
        manifest.update(last_update=update,last_step=step,best_fid=None if best_step is None else best,best_step=best_step)
        if update%200==0: write_json(args.output/'training_manifest.json',manifest)
    manifest['status']='smoke_complete' if args.smoke else 'complete'
    write_json(args.output/'training_manifest.json',manifest)
    if run: run.finish()


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        # Preserve the last valid checkpoint; make failure visible without
        # needing a model to poll the process repeatedly.
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            path=output/'training_manifest.json'
            if path.exists():
                state=json.loads(path.read_text())
                if state.get('pid') == os.getpid():
                    state.update(status='failed',error=repr(error))
                    write_json(path,state)
        raise
