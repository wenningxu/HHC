"""Same-request VQ error attribution, with explicitly labelled GT-oracle controls.

Oracle replacement is a diagnostic, never an inference method or benchmark.
No evaluator feature changes, training updates, or checkpoint writes occur.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_smpl_pair import body192_from_motion
from models.vqvae_interhuman_smpl_raw import NativeSMPLFK
from tools.reconstruct_interhuman_raw_body import load_model
from tools.score_interhuman_vq_geometry import error_sums,finalize
from utils.hhc_frozen_vq import load_frozen_vq,file_sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vq-spec',type=Path,required=True)
    parser.add_argument('--reconstructions',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud GPU only')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);os.nice(10)
    torch.cuda.set_per_process_memory_fraction(.2,0)
    native=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
    net,mean,std,spec=load_frozen_vq(args.vq_spec,'cuda:0',native/'manifest.json')
    data=InterHumanValidationRequestDataset();fk=NativeSMPLFK().cuda();smpl=load_model()
    arms=['predicted','oracle_root_position','oracle_global_rotation','oracle_both_roots','oracle_all_gt']
    totals={a:{} for a in arms};world_joint=world_root=relative_joint=0.;frames=0;cache_max=0.
    started=time.time()
    with h5py.File(native/'motions.h5') as archive,h5py.File(args.reconstructions) as cache,torch.inference_mode():
        assert cache.attrs['status']=='complete' and cache.attrs['vq_checkpoint_sha256']==spec['sha256']
        assert cache.attrs['request_content_sha256']==data.payload['request_sha256']
        for i,r in enumerate(data.requests):
            name,text,a,b,length=data[i];beta=torch.from_numpy(archive[name]['betas'][:]).cuda()
            raw=torch.from_numpy(archive[name]['motion30'][r['crop_start']:r['crop_start']+r['raw_frames']]).cuda()
            x=fk.from_archive(raw,beta);n=len(x);pad=(-n)%4
            z=torch.cat([x,x[-1:].expand(pad,-1,-1)]) if pad else x
            prediction=net((z.reshape(1,-1,276)-mean)/std,beta.reshape(1,20),
                           torch.ones(1,2,dtype=torch.long,device='cuda'))[0]
            pred=(prediction[0,:n]*std+mean).reshape(n,23,12)
            predicted_joints=fk(pred[None],beta.reshape(1,20));true_joints=fk(x[None],beta.reshape(1,20))
            for p,t in zip(predicted_joints,true_joints):
                p,t=p[0,:length],t[0,:length];delta=torch.linalg.vector_norm(p-t,dim=-1)
                world_joint+=float(delta.sum());world_root+=float(delta[:,0].sum())
                relative_joint+=float(torch.linalg.vector_norm((p-p[:,:1])[:,1:]-(t-t[:,:1])[:,1:],dim=-1).sum())
                frames+=length
            for arm in arms:
                value=pred.clone()
                if arm in ('oracle_root_position','oracle_both_roots'):value[:,22]=x[:,22]
                if arm in ('oracle_global_rotation','oracle_both_roots'):value[:,0]=x[:,0]
                if arm=='oracle_all_gt':value=x
                body=body192_from_motion(fk.to_archive(value,beta).cpu(),beta.cpu(),smpl).numpy()
                q=data.adapter.process_loaded_pair(*body,anchor_person=r['anchor_person'],
                    output_person_swap=r['output_person_swap'],normalize=True)
                for target,p,key in zip((a,b),q,('motion1','motion2')):
                    block=torch.zeros_like(target);block[:length]=torch.from_numpy(p)
                    if arm=='predicted':cache_max=max(cache_max,float(np.abs(p-cache[key][i,:length]).max()))
                    sums=error_sums(target[None],block[None],torch.tensor([length]),data.adapter.mean,data.adapter.std)
                    for key,value_sum in sums.items():totals[arm][key]=totals[arm].get(key,0)+value_sum
            if (i+1)%96==0:print(json.dumps({'completed':i+1,'elapsed_s':time.time()-started}),flush=True)
    assert cache_max==0.,'Diagnostic must reproduce the frozen prediction cache exactly'
    result={'schema':'hhc_native_vq_root_attribution_v1','status':'complete','samples':480,
        'vq_step':spec['step'],'vq_sha256':spec['sha256'],'selection_scope':spec['selection_scope'],
        'is_benchmark_result':False,'oracle_arms_use_gt':arms[1:],
        'interpretation':'Ground-truth replacement only localizes early VQ error; no oracle is a deployable method or claimed improvement.',
        'same_request_native_world':{'full_mpjpe_m':world_joint/(frames*22),'root_position_error_m':world_root/frames,
                                    'root_relative_body_mpjpe_m':relative_joint/(frames*21),'person_frames':frames},
        'official_processed_arms':{arm:finalize(sums) for arm,sums in totals.items()},
        'prediction_cache_max_normalized_abs':cache_max,'request_sha256':data.payload['request_sha256'],
        'reconstructions_sha256':file_sha(args.reconstructions),'script_sha256':file_sha(Path(__file__)),
        'elapsed_s':time.time()-started,'gpu_peak_gib':torch.cuda.max_memory_allocated()/1024**3}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
