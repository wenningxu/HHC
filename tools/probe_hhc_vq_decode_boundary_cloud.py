"""Val-only decoder edge-context probe; all candidate arms are GT-free.

Repeat 0/8/16 first/last *predicted* tokens before decoding and trim that
context. Ground truth is used only for scoring. Does not modify the VQ or MGT.
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
from tools.score_interhuman_vq_geometry import error_sums,finalize,physical_sums,physical_metrics
from utils.hhc_frozen_vq import load_frozen_vq,file_sha
from utils.rotation_conversions import rotation_6d_to_matrix


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vq-spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int,choices=[32,480],default=32)
    args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud GPU only')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);os.nice(10)
    torch.cuda.set_per_process_memory_fraction(.2,0)
    native=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
    net,mean,std,spec=load_frozen_vq(args.vq_spec,'cuda:0',native/'manifest.json')
    before={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    data=InterHumanValidationRequestDataset();fk=NativeSMPLFK().cuda();smpl=load_model()
    totals={k:{} for k in ('full_forward','decode0','decode8','decode16')}
    physics={k:{} for k in totals};angles={k:{'first_sum_deg':0.,'first_count':0,'interior_sum_deg':0.,'interior_count':0} for k in totals}
    chosen=list(range(480)) if args.limit==480 else np.linspace(0,479,32,dtype=int).tolist()
    started=time.time()
    with h5py.File(native/'motions.h5') as archive,torch.inference_mode():
        for position,i in enumerate(chosen):
            r=data.requests[i];name,_,a,b,length=data[i];beta=torch.from_numpy(archive[name]['betas'][:]).cuda()
            raw=torch.from_numpy(archive[name]['motion30'][r['crop_start']:r['crop_start']+r['raw_frames']]).cuda()
            x=fk.from_archive(raw,beta);n=len(x);pad=(-n)%4
            z=torch.cat([x,x[-1:].expand(pad,-1,-1)]) if pad else x
            normalized=(z.reshape(1,-1,276)-mean)/std;gender=torch.ones(1,2,dtype=torch.long,device='cuda')
            full=net(normalized,beta.reshape(1,20),gender)[0]
            tokens=net(normalized,beta.reshape(1,20),gender,type='encode')
            true_rot=rotation_6d_to_matrix(x[:length,0].reshape(length,2,6))
            for arm in totals:
                if arm=='full_forward':prediction=full[0,:n]
                else:
                    k=int(arm[6:]);codes=tokens
                    if k:codes=torch.cat([codes[:,:1].expand(-1,k,-1),codes,codes[:,-1:].expand(-1,k,-1)],dim=1)
                    prediction=net(codes,beta.reshape(1,20),gender,type='decode')[0,k*4:k*4+n]
                pred=(prediction*std+mean).reshape(n,23,12)
                rot=rotation_6d_to_matrix(pred[:length,0].reshape(length,2,6))
                trace=(rot.transpose(-1,-2)@true_rot).diagonal(dim1=-2,dim2=-1).sum(-1)
                error=torch.acos(((trace-1)/2).clamp(-1,1))*180/torch.pi
                angles[arm]['first_sum_deg']+=float(error[0].sum());angles[arm]['first_count']+=2
                angles[arm]['interior_sum_deg']+=float(error[1:].sum());angles[arm]['interior_count']+=2*(length-1)
                body=body192_from_motion(fk.to_archive(pred,beta).cpu(),beta.cpu(),smpl).numpy()
                q=data.adapter.process_loaded_pair(*body,anchor_person=r['anchor_person'],output_person_swap=r['output_person_swap'],normalize=True)
                blocks=[]
                for target,p in zip((a,b),q):
                    block=torch.zeros_like(target);block[:length]=torch.from_numpy(p);blocks.append(block)
                    sums=error_sums(target[None],block[None],torch.tensor([length]),data.adapter.mean,data.adapter.std)
                    for key,value in sums.items():totals[arm][key]=totals[arm].get(key,0)+value
                sums=physical_sums(a[None],b[None],blocks[0][None],blocks[1][None],torch.tensor([length]),data.adapter.mean,data.adapter.std,.1)
                for key,value in sums.items():physics[arm][key]=physics[arm].get(key,0)+value
            if (position+1)%32==0:print(json.dumps({'completed':position+1,'elapsed_s':time.time()-started}),flush=True)
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in net.state_dict().items())
    result={'schema':'hhc_native_vq_decoder_context_probe_v1','status':'complete','samples':len(chosen),'indices':chosen,
        'vq_step':spec['step'],'vq_sha256':spec['sha256'],'request_sha256':data.payload['request_sha256'],
        'protocol':'fixed val subset; native reconstruction; predicted token edge repeat; no GT replacements; physical contact is geometry proxy',
        'is_generation_evaluation':False,'is_formal_test':False,'fid':'not computed',
        'arms':{k:{'geometry':finalize(totals[k]),'physical':physical_metrics(physics[k]),
            'global_root_rotation_first_deg':angles[k]['first_sum_deg']/angles[k]['first_count'],
            'global_root_rotation_interior_deg':angles[k]['interior_sum_deg']/angles[k]['interior_count']} for k in totals},
        'frozen_vq_unchanged':True,'script_sha256':file_sha(Path(__file__)),'elapsed_s':time.time()-started,
        'gpu_peak_gib':torch.cuda.max_memory_allocated()/1024**3}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
