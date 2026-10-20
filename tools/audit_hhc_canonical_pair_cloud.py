"""Real 480-request coordinate parity plus independent CPU gradient checks."""
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
from utils.hhc_frozen_vq import load_frozen_vq,file_sha
from utils.hhc_canonical_pair import canonical_pair_joints,canonical_pair_loss


def cpu_checks():
    generator=torch.Generator().manual_seed(20261016)
    target=torch.randn(1,7,2,22,3,generator=generator,dtype=torch.float64)
    target[:,2,:,5,1]=-10 # unique floors: directional finite difference stays smooth
    predicted=(target+.05*torch.randn(target.shape,generator=generator,dtype=torch.float64)).requires_grad_()
    loss=canonical_pair_loss(predicted,target);gradient=torch.autograd.grad(loss,predicted)[0]
    direction=torch.randn(target.shape,generator=generator,dtype=torch.float64);direction/=direction.norm()
    epsilon=1e-6
    numerical=float((canonical_pair_loss(predicted.detach()+epsilon*direction,target)-canonical_pair_loss(predicted.detach()-epsilon*direction,target))/(2*epsilon))
    analytical=float((gradient*direction).sum());assert abs(numerical-analytical)<1e-7
    theta=target.new_tensor(.83);c,s=theta.cos(),theta.sin()
    rotation=torch.stack([torch.stack([c,c*0,s]),torch.tensor([0.,1.,0.],dtype=torch.float64),torch.stack([-s,s*0,c])])
    moved=target@rotation.T+target.new_tensor([3.,5.,-2.])
    gauge_max=max(float((canonical_pair_joints(target,a)-canonical_pair_joints(moved,a)).abs().max()) for a in (1,2))
    assert gauge_max<1e-11 and torch.isfinite(gradient).all() and float(gradient.norm())>0
    assert torch.equal(canonical_pair_joints(target,1,True),canonical_pair_joints(target,1)[:,:,[1,0]])
    bad=target.clone();bad[:,:,0,2]=bad[:,:,0,1]
    try:canonical_pair_joints(bad)
    except ValueError:rejected=True
    else:raise AssertionError('Degenerate hip facing accepted')
    return {'directional_derivative':analytical,'finite_difference':numerical,'absolute_gradient_error':abs(numerical-analytical),
            'gauge_invariance_max_m':gauge_max,'swap_exact':True,'degenerate_hips_rejected':rejected,'device':'cpu'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vq-spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('GPU inference only on cloud')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);os.nice(10)
    checks=cpu_checks();torch.cuda.set_per_process_memory_fraction(.2,0)
    native=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
    net,mean,std,spec=load_frozen_vq(args.vq_spec,'cuda:0',native/'manifest.json')
    data=InterHumanValidationRequestDataset();fk=NativeSMPLFK().cuda();smpl=load_model();started=time.time()
    maxima={'GT':0.,'predicted':0.};mean_sum={'GT':0.,'predicted':0.};counts={'GT':0,'predicted':0};worst=[]
    with h5py.File(native/'motions.h5') as archive,torch.inference_mode():
        for i,r in enumerate(data.requests):
            group=archive[r['identifier']];n=r['raw_frames'];start=r['crop_start']
            beta=torch.from_numpy(group['betas'][:]).cuda();raw=torch.from_numpy(group['motion30'][start:start+n]).cuda()
            x=fk.from_archive(raw,beta);pad=(-n)%4;z=torch.cat([x,x[-1:].expand(pad,-1,-1)]) if pad else x
            prediction=net((z.reshape(1,-1,276)-mean)/std,beta.reshape(1,20),torch.ones(1,2,dtype=torch.long,device='cuda'))[0]
            pred=(prediction[0,:n]*std+mean).reshape(n,23,12)
            for label,value in (('GT',x),('predicted',pred)):
                joints=torch.stack(fk(value[None],beta.reshape(1,20)),dim=2)
                body=body192_from_motion(fk.to_archive(value,beta).cpu(),beta.cpu(),smpl).numpy()
                for anchor in (1,2):
                    for swap in (False,True):
                        actual=canonical_pair_joints(joints,anchor,swap)[0].cpu().numpy()
                        official=data.adapter.process_loaded_pair(*body,anchor_person=anchor,output_person_swap=swap,normalize=False)
                        expected=np.stack([v[:,:66].reshape(-1,22,3) for v in official],axis=1)
                        error=np.abs(actual-expected);maximum=float(error.max())
                        maxima[label]=max(maxima[label],maximum);mean_sum[label]+=float(error.sum());counts[label]+=error.size
                        worst.append({'id':r['identifier'],'label':label,'anchor':anchor,'swap':swap,'max_abs_m':maximum})
            if (i+1)%96==0:print(json.dumps({'processed':i+1,'max_abs_m':maxima,'elapsed_s':time.time()-started}),flush=True)
    result={'status':'passed' if max(maxima.values())<=1e-4 else 'failed','samples':480,
        'cases':480*2*2*2,'max_coordinate_abs_m':maxima,'mean_coordinate_abs_m':{k:mean_sum[k]/counts[k] for k in counts},
        'protocol':'Both anchors and output swaps; native Y-up FK vs released process_loaded_pair position channels; drop last raw frame',
        'cpu_checks':checks,'worst':sorted(worst,key=lambda r:r['max_abs_m'],reverse=True)[:8],
        'vq_step':spec['step'],'vq_sha256':spec['sha256'],'request_sha256':data.payload['request_sha256'],
        'sources':{p:file_sha(ROOT/p) for p in ('tools/audit_hhc_canonical_pair_cloud.py','utils/hhc_canonical_pair.py')},
        'official_preprocessing_sha256':file_sha(data.adapter.intermask_root/'data/utils.py'),
        'elapsed_s':time.time()-started,'gpu_peak_gib':torch.cuda.max_memory_allocated()/1024**3,
        'scope':'Numerical coordinate and gradient equivalence only, no training/quality improvement claim'}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert result['status']=='passed','Coordinate parity failed; do not train this loss'


if __name__=='__main__':main()
