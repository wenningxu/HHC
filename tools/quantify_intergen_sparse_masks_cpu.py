"""Real-request CPU audit of sparse distance/contact reductions."""
import os, json, hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
torch.set_num_threads(1)
from dataset.interhuman_root_rotation import RootRotationData
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from tools.intergen_rotation_grouped_objective import rotation_objective as old_objective
from tools.intergen_rotation_objective_sparse_v2 import rotation_objective as new_objective

def main():
    root=Path('/workspace/HHC'); out=root/'output/hhc_sota_20261016/intergen_sparse_masks_actual_cpu_v2_20261017.json'
    files=sorted((root/'output/intergen_heads_bug_diagnostic_20261017').glob('step103000_request*.npz'))
    assert len(files)==8, f'expected 8 NPZ, found {len(files)}'
    data=RootRotationData(split='val'); req=InterHumanValidationRequestDataset(adapter=data.adapter)
    z=np.load(root/'output/intergen_rotation_scratch_runpod_20261015/normalization.npz')
    mean=torch.from_numpy(z['mean']); std=torch.from_numpy(z['std'])
    rows=[]
    for i,p in enumerate(files):
        q=np.load(p,allow_pickle=False); keys=list(q.keys())
        g=None
        for k in ('generated268','generated','gen','prediction','pred','output'):
            if k in q: g=q[k]; break
        assert g is not None, (str(p),keys)
        g=np.asarray(g)
        if g.ndim==2: g=g[None]
        elif g.ndim==3 and g.shape[-2:]==(2,268): g=g[None]
        request_index=int(p.stem.split('request')[-1])
        request=req.requests[request_index]
        name,caption,a,b,length=req[request_index]
        raw_np, offsets_np=data.pair(request['identifier'],int(request['crop_start']),False,int(request['anchor_person']),bool(request['output_person_swap']))
        raw_np=raw_np[:length]
        target_raw=torch.from_numpy(raw_np).float()[None]
        assert torch.allclose(target_raw[...,:262],torch.stack((a,b),1)[:length].float()[None]*std[:262]+mean[:262],atol=2e-5)
        target=(target_raw-mean)/std
        pred_raw=torch.from_numpy(g).float()
        if pred_raw.shape[-1]==268: pred_raw=pred_raw[:,:length]
        else: raise ValueError((p, g.shape))
        pred=(pred_raw-mean)/std
        assert torch.allclose(pred*std+mean,pred_raw,atol=1e-5)
        offsets=torch.from_numpy(offsets_np).float()[None]
        lengths=torch.tensor([length]); t=torch.zeros(1,dtype=torch.long)
        args=(offsets,lengths,t,mean,std)
        old=old_objective(pred,target,*args,reference_micro_batch=1)
        new=new_objective(pred,target,*args,reference_micro_batch=1)
        raw=target_raw
        positions=raw[...,:66].reshape(1,length,2,22,3)
        d=torch.cdist(positions[:,:,0],positions[:,:,1]); k1=int((d<1).sum()); k2=int((d<.1).sum())
        c=int((raw[:,:,:,258:262]>0).sum())
        rows.append({'file':str(p),'keys':keys,'length':int(length),'k_lt1':k1,'k_lt01':k2,'contact_count':c,'old':{k:float(v) for k,v in old.items()},'new':{k:float(v) for k,v in new.items()},'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    result={'schema':'intergen_sparse_masks_actual_cpu_v2','status':'complete','cuda_initialized':torch.cuda.is_initialized(),'rows':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limits':'Actual eight diagnostic requests; physical generated268 converted to normalized space; request-specific crop/anchor/output swap replayed; denominator audit only.'}
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
