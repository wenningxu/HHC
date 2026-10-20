"""Decoder-only canonical-pair loss pilot against the completed equal-budget control.

Both arms freeze encoder/codebooks/shape condition and use exactly the same
training windows. Canonical joint SmoothL1 uses weight 3, matching the existing native joint
coefficient. Both anchors are averaged; source parity is required before training. No validation or test training.
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import default_collate
from dataset.interhuman_interx_raw import InterHumanInterXRaw
from models.vqvae_interhuman_smpl_raw import NativeSMPLFK,RawLoss,RawNormalizer
from utils.hhc_frozen_vq import load_frozen_vq,file_sha
from utils.hhc_resumable_batches import ResumableWindowBatches
from utils.rotation_conversions import rotation_6d_to_matrix
from utils.hhc_canonical_pair import canonical_pair_loss


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def root_geodesic(pred,target):
    p=rotation_6d_to_matrix(pred[:,:,0].reshape(-1,6))
    t=rotation_6d_to_matrix(target[:,:,0].reshape(-1,6))
    cosine=((t@p.transpose(-1,-2)).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return torch.acos(cosine.clamp(-1+1e-7,1-1e-7)).mean()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vq-spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--reference-control',type=Path,required=True);args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('GPU training is cloud only')
    assert args.updates==1000,'Preregistered short equal-budget pilot'
    parity_path=ROOT/'output/hhc_sota_20261016/canonical_pair_parity.json'
    parity=json.loads(parity_path.read_text());assert parity['status']=='passed'
    assert parity['sources']['utils/hhc_canonical_pair.py']==file_sha(ROOT/'utils/hhc_canonical_pair.py')
    reference=json.loads(args.reference_control.read_text())
    assert reference['arm']=='decoder_control' and reference['decoder_updates']==1000 and reference['root_loss_weight']==0
    assert file_sha(reference['checkpoint'])==reference['sha256']
    args.output=args.output.resolve();assert args.output.is_relative_to(ROOT/'output')
    args.output.mkdir(exist_ok=args.resume)
    lock=(args.output/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=args.output/'STATUS.json';state={'status':'initializing','pid':os.getpid(),'started':time.time(),'arms':{}}
    torch.set_num_threads(2);torch.set_num_interop_threads(2);os.nice(10)
    torch.cuda.set_per_process_memory_fraction(.2,0)
    native=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
    sources=['tools/train_hhc_decoder_canonical_pilot_cloud.py','tools/train_hhc_decoder_root_pilot_cloud.py','utils/hhc_canonical_pair.py','utils/hhc_frozen_vq.py','utils/hhc_resumable_batches.py',
             'models/vqvae_interhuman_smpl_raw.py','dataset/interhuman_interx_raw.py','utils/losses.py','utils/rotation_conversions.py']
    contract={'schema':'hhc_decoder_canonical_equal_budget_pilot_v1','base_spec_sha256':file_sha(args.vq_spec),
        'data_sha256':file_sha(native/'manifest.json'),'parity_sha256':file_sha(parity_path),
        'reference_control_spec_sha256':file_sha(args.reference_control),'updates':1000,'batch':256,'seed':20261016,
        'window':64,'precision':'fp32','trainable':'decoder only; all other modules fixed eval',
        'optimizer':{'lr':.0002,'betas':[.9,.99],'weight_decay':.00002,'scheduler':None},
        'arms':{'decoder_canonical_pair':3.},
        'loss':'original RawLoss sum + .02 constant frozen-encoder commitment + arm_weight * mean_both_anchor_canonical_joint_SmoothL1',
        'sources':{p:file_sha(ROOT/p) for p in sources},'local_checkpoint_downloads':False,
        'scope':'early 5000-step VQ canonical-pair pilot; matched to previous decoder_control1000, not final VQ/MGT or SOTA'}
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if args.resume:
        assert json.loads((args.output/'contract.json').read_text())==contract
        state=json.loads(status.read_text());state.update(pid=os.getpid(),status='resuming')
    else:write(args.output/'contract.json',contract)
    state['contract_sha256']=digest;write(status,state)
    try:
        initial,mean_cpu,std_cpu,spec=load_frozen_vq(args.vq_spec,'cpu',native/'manifest.json')
        assert reference['base_vq_sha256']==spec['sha256'] and reference['base_vq_step']==spec['step']
        del initial
        fk_cpu=NativeSMPLFK();data=InterHumanInterXRaw(native,'train',fk_cpu,stats=(mean_cpu,std_cpu))
        fk=fk_cpu.cuda()
        for arm,weight in contract['arms'].items():
            directory=args.output/arm
            if state['arms'].get(arm,{}).get('status')=='complete':
                receipt=json.loads((directory/'decoder_spec.json').read_text())
                assert file_sha(receipt['checkpoint'])==receipt['sha256'];continue
            directory.mkdir(exist_ok=args.resume)
            random.seed(20261016);np.random.seed(20261016);torch.manual_seed(20261016)
            net,mean,std,_=load_frozen_vq(args.vq_spec,'cuda:0',native/'manifest.json')
            frozen={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if not k.startswith('decoder.')}
            net.decoder.requires_grad_(True).train()
            params=list(net.decoder.parameters())
            optimizer=torch.optim.AdamW(params,lr=.0002,betas=(.9,.99),weight_decay=.00002)
            objective=RawLoss(SimpleNamespace(motion_rep='raw'),fk,RawNormalizer(mean,std,'cuda:0'))
            batches=ResumableWindowBatches(len(data),256,20261016);start=0;order=hashlib.sha256()
            if args.resume and (directory/'latest.pth').exists():
                receipt=json.loads((directory/'latest.receipt.json').read_text());assert file_sha(directory/'latest.pth')==receipt['sha256']
                saved=torch.load(directory/'latest.pth',map_location='cpu',weights_only=False)
                assert saved['contract_sha256']==digest and saved['arm']==arm
                net.decoder.load_state_dict(saved['decoder'],strict=True);optimizer.load_state_dict(saved['optimizer'])
                start=saved['update'];batches.load_state_dict(saved['batches'])
                # Reconstruct the digest of prior batches without consuming the live cursor.
                replay=ResumableWindowBatches(len(data),256,20261016)
                for _ in range(start):order.update(np.asarray(replay.draw(),dtype=np.int64).tobytes())
                random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
                torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            state.update(status='training',arm=arm,update=start,trainable_parameters=sum(p.numel() for p in params),
                         base_vq_step=spec['step'],base_vq_sha256=spec['sha256'])
            write(status,state);started=time.time()
            for update in range(start+1,1001):
                indices=batches.draw();order.update(np.asarray(indices,dtype=np.int64).tobytes())
                x,beta,gender=default_collate([data[i] for i in indices]);x,beta,gender=x.cuda(),beta.cuda(),gender.cuda()
                prediction,commit,perplexity=net(x,beta,gender)
                components=objective(prediction,x,beta)
                root=root_geodesic(objective.motion_pred_de,objective.motion_gt_de)
                canonical=canonical_pair_loss(torch.stack([objective.motion_pred_de_joints_p1,objective.motion_pred_de_joints_p2],dim=2),
                    torch.stack([objective.motion_gt_de_joints_p1,objective.motion_gt_de_joints_p2],dim=2))
                loss=sum(components.values())+.02*commit+0.*root+weight*canonical
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True);optimizer.step()
                if update<=4 or update%100==0:
                    assert all(torch.equal(v.detach().cpu(),frozen[k]) for k,v in net.state_dict().items() if k in frozen)
                    assert all(p.grad is None for name,p in net.named_parameters() if not name.startswith('decoder.'))
                    row={'arm':arm,'update':update,'loss':float(loss.detach()),'root_geodesic_rad':float(root.detach()),
                         'root_weight':0.,'canonical_weight':weight,'canonical_joint_loss':float(canonical.detach()),'grad_norm':float(norm),'elapsed_s':time.time()-started,
                         'gpu_peak_gib':torch.cuda.max_memory_allocated()/1024**3}
                    with (directory/'training.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True);state.update(update=update);write(status,state)
                if update%250==0:
                    payload={'decoder':net.decoder.state_dict(),'optimizer':optimizer.state_dict(),'batches':batches.state_dict(),
                             'update':update,'arm':arm,'contract_sha256':digest,'python_rng':random.getstate(),
                             'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
                    pending=directory/'latest.tmp';torch.save(payload,pending);pending.replace(directory/'latest.pth')
                    write(directory/'latest.receipt.json',{'sha256':file_sha(directory/'latest.pth'),'update':update,'cloud_only':True})
            checkpoint=directory/'decoder.pth'
            torch.save({'decoder':net.decoder.state_dict(),'base_vq_sha256':spec['sha256'],'updates':1000,
                        'arm':arm,'contract_sha256':digest},checkpoint)
            result={'schema':'hhc_decoder_only_override_v1','checkpoint':str(checkpoint),'sha256':file_sha(checkpoint),
                'bytes':checkpoint.stat().st_size,'base_vq_sha256':spec['sha256'],'base_vq_step':spec['step'],
                'decoder_updates':1000,'arm':arm,'root_loss_weight':0.,'canonical_pair_loss_weight':weight,'contract_sha256':digest,
                'frozen_non_decoder_unchanged':True,'batch_order_sha256':order.hexdigest(),'is_quality_evaluation':False}
            assert result['batch_order_sha256']==reference['batch_order_sha256']
            write(directory/'decoder_spec.json',result);state['arms'][arm]={'status':'complete',**result};write(status,state)
            del prediction,components,root,canonical,loss,optimizer,objective,params,net
            torch.cuda.empty_cache()
        assert len({r['batch_order_sha256'] for r in state['arms'].values()})==1
        for p,expected in contract['sources'].items():assert file_sha(ROOT/p)==expected
        state.update(status='complete_training_only',ended=time.time());write(status,state)
    except Exception as error:
        state.update(status='failed',error=repr(error),ended=time.time());write(status,state);raise


if __name__=='__main__':main()
