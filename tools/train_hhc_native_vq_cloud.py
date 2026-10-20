"""Cloud-only native HHC VQ rebuild with complete, contract-bound recovery.

This is a fresh branch, not a recovery of the unavailable historical VQ185k.
Original model, loss, warmup, scheduler and validation equations are retained.
"""
import argparse
import fcntl
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
from torch.utils.data import default_collate

from dataset.interhuman_interx_raw import InterHumanInterXRaw
from dataset.interhuman_smpl_pair import to_interhuman
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from models.vqvae_interhuman_smpl_raw import NativeSMPLFK,VQSMPLRaw,RawNormalizer,RawLoss
from models.interhuman_evaluator_wrapper import InterHumanEvaluatorWrapper,load_official_metrics_module
from tools.reconstruct_interhuman_raw_body import load_model
from tools.score_fair_interhuman_main import main_metrics_from_embeddings
from utils.quantizer_state import quantizer_training_state,restore_quantizer_training_state
from utils.runtime import evaluating,preserving_rng_state
from utils.hhc_resumable_batches import ResumableWindowBatches


def write_json(path,data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


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
    p.add_argument('--eval-device',default='cpu')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--main-steps',type=int,default=300000)
    p.add_argument('--schedule-steps',type=int,default=300000)
    p.add_argument('--stop-after-update',type=int)
    p.add_argument('--gpu-memory-fraction',type=float,default=.2)
    p.add_argument('--wandb-mode',choices=['disabled','online','offline'],default='disabled')
    p.add_argument('--seed',type=int,default=123)
    args=p.parse_args()
    if str(ROOT)!='/workspace/HHC' or args.device!='cuda:0':
        raise RuntimeError('This entrypoint only runs on the attributed cloud GPU')
    if not 0<args.gpu_memory_fraction<=.25:raise ValueError('Limit VQ process to at most 25% GPU memory')
    if not 0<args.main_steps<=args.schedule_steps:raise ValueError('Invalid training horizon')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    os.nice(10)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,0)
    args.nb_code=12288; args.mu=.99; args.motion_rep='raw'
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    args.output=args.output.resolve()
    if not args.output.is_relative_to(ROOT/'output'):raise ValueError('Run directory must be inside cloud output')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    training_lock=(args.output/'training.lock').open('a')
    fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if shutil.disk_usage(args.output).free<5*1024**3:
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
    batches=ResumableWindowBatches(len(train),256,args.seed)
    net=VQSMPLRaw(args).to(args.device)
    fk=fk.to(args.device)
    objective=RawLoss(args,fk,RawNormalizer(train.mean,train.std,args.device))
    optimizer=torch.optim.AdamW(net.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=2e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.schedule_steps,eta_min=0)
    sources=['tools/train_hhc_native_vq_cloud.py','train/train_vq_interhuman_smpl_raw.py','models/vqvae_interhuman_smpl_raw.py',
             'dataset/interhuman_interx_raw.py','models/encdec.py','models/quantize_cnn.py',
             'models/vqvae_sep3_shape.py','utils/losses.py','dataset/interhuman_smpl_pair.py',
             'models/interhuman_evaluator_wrapper.py','utils/quantizer_state.py','utils/hhc_resumable_batches.py','utils/runtime.py']
    contract={'schema':'hhc_native_vq_cloud_rebuild_v1','origin':'fresh_cloud_rebuild_not_historical185k',
              'seed':args.seed,'data':str(args.data),
              'parameters':sum(x.numel() for x in net.parameters()),'batch':256,'window':64,
              'codebook_size':12288,'latent_dims':[96,64],'shape_condition_dim':8,
              'precision':'fp32','lr':.0002,'weight_decay':.00002,'adam_betas':[.9,.99],
              'warmup_updates':999,'main_steps':args.main_steps,'schedule_steps':args.schedule_steps,'eval_interval':5000,
              'checkpoint_interval':1000,'eval_device':args.eval_device,'gpu_memory_fraction':args.gpu_memory_fraction,
              'batch_order':'independent CPU Generator random-permutation, drop_last; full permutation/cursor serialized; differs from historical RandomSampler exact order',
              'wandb_mode':args.wandb_mode,
              'loss_weights':{'main':1,'Joint':3,'Velocity':100,'Rotation':.01,'Root':.1,'FootContact':10,'DistanceMap':.1,'commit':.02},
              'sources':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
              'dataset_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
              'notes':['Original Inter-X raw slots, no hand channel; no speed features as model inputs.',
                       'Y-up global basis and pelvis root; invertible to native SMPL with original beta.',
                       'SMPL beta-dependent FK replaces Inter-X group-specific SMPL-X skeleton.',
                       'Only finite empty-contact safeguard differs from historical loss reduction.',
                       'Fixed InterHuman val monitor; not Inter-X evaluator or formal official test.',
                       'Fresh cloud rebuild. Historical VQ185k weights are unavailable; no pretrained generator weight is read.',
                       'Same historical window distribution; resumable batch order differs. Not a bitwise historical reproduction.',
                       'All model/optimizer/EMA/RNG/batch cursor recovery stays cloud-only.',
                       'Checkpoint retention: latest, best val FID, step175000 only; do not accumulate milestones.']}
    contract_hash=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    contract_path=args.output/'training_contract.json'
    if args.resume:assert json.loads(contract_path.read_text())==contract,'Resume contract mismatch'
    else:write_json(contract_path,contract)
    manifest={'status':'initializing','pid':os.getpid(),'contract':contract,'contract_sha256':contract_hash,
              'local_checkpoint_downloads':False,'is_historical_vq185k':False}
    prior=json.loads((args.output/'training_manifest.json').read_text()) if args.resume else {}
    run=None
    if args.wandb_mode!='disabled':
        import wandb
        run=wandb.init(project='NewInter',group='vq_interhuman_smpl_raw',name=args.output.name,
                       dir=str(args.output),mode=args.wandb_mode,config=contract,
                       id=prior.get('wandb_run_id'),resume='allow')
        manifest['wandb_run_id']=run.id
    write_json(args.output/'training_manifest.json',manifest)
    log({'event':'ready','windows':len(train),'train_sequences':len(train.ids),'val_sequences':len(val.ids),
                 'parameters':contract['parameters']})
    evaluator=InterHumanEvaluatorWrapper.from_official_checkpoint(args.eval_device)
    metrics=load_official_metrics_module()
    adapter=OfficialInterHumanAdapter()
    smpl=load_model()
    best=float('inf'); best_step=None
    start=0
    if args.resume:
        receipt=json.loads((args.output/'latest.receipt.json').read_text())
        assert receipt['sha256']==sha(args.output/'latest.pth')
        saved=torch.load(args.output/'latest.pth',map_location='cpu',weights_only=False)
        assert saved['contract_sha256']==contract_hash
        assert torch.equal(saved['mean'],train.mean) and torch.equal(saved['std'],train.std)
        net.load_state_dict(saved['net'],strict=True)
        assert not restore_quantizer_training_state(net,saved['quantizer_training_state'])
        assert all(torch.equal(v.detach().cpu(),saved['net'][k]) for k,v in net.state_dict().items())
        for name,values in quantizer_training_state(net).items():
            for key,value in values.items():
                expected=saved['quantizer_training_state'][name][key]
                assert torch.equal(value.cpu(),expected) if torch.is_tensor(value) else value==expected
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        batches.load_state_dict(saved['batch_order'])
        start=saved['update'];best=saved['best_fid'];best_step=saved['best_step']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        log({'event':'resumed','update':start,'next_update':start+1,'batch_epoch':batches.epoch,'batch_cursor':batches.cursor,
             'recovery_sha256':receipt['sha256'],'optimizer_state_count':len(optimizer.state),
             'model_and_ema_exactly_reloaded':True,'batch_order_restored':True})
        del saved
    manifest.update(status='running',last_update=start,last_step=max(0,start-999),best_fid=None if best_step is None else best,best_step=best_step)
    write_json(args.output/'training_manifest.json',manifest)
    started=time.monotonic()
    end=min(args.main_steps+999,args.stop_after_update) if args.stop_after_update is not None else args.main_steps+999
    if end<=start:raise ValueError('Stop update must exceed resumed update')
    for update in range(start+1,end+1):
        warmup=update<=999
        step=0 if warmup else update-999
        if warmup:
            for group in optimizer.param_groups: group['lr']=.0002*(update+1)/1001
        x,beta,gender=default_collate([train[index] for index in batches.draw()])
        x,beta,gender=x.to(args.device),beta.to(args.device),gender.to(args.device)
        pred,commit,perplexity=net(x,beta,gender)
        components=objective(pred,x,beta)
        loss=sum(components.values())+.02*commit
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss update{update}: {components}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update<=8 or update%50==0:
            if any(not torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None):
                raise FloatingPointError(f'Nonfinite gradient update{update}')
        optimizer.step()
        if not warmup: scheduler.step()
        if update<=8 or update%200==0:
            row={'event':'train','update':update,'step':step,'warmup':warmup,'loss':loss.item(),
                 'lr':optimizer.param_groups[0]['lr'],'perplexity':perplexity.tolist(),
                 'elapsed_s':time.monotonic()-started,'gpu_peak_gib':torch.cuda.max_memory_allocated(args.device)/1024**3,
                 **{k:v.item() for k,v in components.items()},'commit':commit.item()}
            log(row)
            if run: wandb.log({k:v for k,v in row.items() if isinstance(v,(int,float))},step=update)
        should_eval=not warmup and step%5000==0
        if should_eval:
            result=validate(net,val,fk,smpl,adapter,evaluator,metrics,args.device,count=480)
            is_best=result['fid']<best
            if is_best: best,best_step=result['fid'],step
            log({'event':'validation','step':step,'best_fid':best,'best_step':best_step,**result})
            write_json(args.output/f'validation_{step:06d}.json',result)
            if run: wandb.log({'val/'+k:v for k,v in result.items() if isinstance(v,(int,float))},step=update)
        else: is_best=False
        if update in (4,999,end) or (step>0 and step%1000==0):
            if shutil.disk_usage(args.output).free<2*1024**3:
                raise RuntimeError('Disk below 2 GiB: stop before overwriting last valid checkpoint')
            state={'net':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'quantizer_training_state':quantizer_training_state(net),'update':update,'step':step,
                   'best_fid':best,'best_step':best_step,'mean':train.mean,'std':train.std,
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'manifest':manifest,
                   'batch_order':batches.state_dict(),'contract_sha256':contract_hash}
            pending=args.output/'latest.tmp'
            torch.save(state,pending); pending.replace(args.output/'latest.pth')
            write_json(args.output/'latest.receipt.json',{'update':update,'step':step,'bytes':(args.output/'latest.pth').stat().st_size,
                'sha256':sha(args.output/'latest.pth'),'contract_sha256':contract_hash,'cloud_only':True})
            for wanted,name in ((is_best,'best_fid.pth'),(step==175000,'step175000.pth')):
                if wanted:
                    torch.save({'net':net.state_dict(),'step':step,'mean':train.mean,'std':train.std,'manifest':manifest},pending)
                    pending.replace(args.output/name)
        manifest.update(last_update=update,last_step=step,best_fid=None if best_step is None else best,best_step=best_step)
        if update<=8 or update%200==0: write_json(args.output/'training_manifest.json',manifest)
    manifest['status']='complete' if end==args.main_steps+999 else 'checkpoint_stop'
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
