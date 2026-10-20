"""Cloud GPU diagnostic: retain root6 and measure direct-position vs FK outputs."""
from pathlib import Path
import argparse,hashlib,json,os,random,sys,time
import numpy as np
import torch
import yaml
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset.interhuman_root_rotation import RootRotationData,fk_positions,PARENTS
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=8);args=p.parse_args()
 assert str(ROOT)=='/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 control=ROOT/'output/intergen_official_baseline_transition_control_20261017'
 assert json.loads((control/'STATUS.json').read_text())['status']=='paused_by_user_for_official_baseline'
 out=ROOT/'output/intergen_heads_bug_diagnostic_20261017';out.mkdir(exist_ok=False)
 (out/'STATUS.json').write_text(json.dumps({'status':'preparing','pid':os.getpid()})+'\n')
 torch.set_num_threads(2)
 data=RootRotationData();requests=InterHumanValidationRequestDataset(adapter=data.adapter)
 source=ROOT/'third_party/intergen_base_20261008';os.chdir(source);sys.path.insert(0,str(source))
 from models.intergen import InterGen
 cfg=SimpleNamespace(**yaml.safe_load((source/'configs/model.yaml').read_text()));cfg.INPUT_DIM=268
 model=InterGen(cfg);model.positional_embedding.requires_grad_(False)
 names={n for n,v in model.named_parameters() if v.requires_grad}
 model.to('cuda').eval()
 mean_rest=torch.from_numpy(data.rest0.copy())
 offsets=mean_rest.clone();offsets[1:]-=mean_rest[torch.tensor(PARENTS[1:])];offsets[0]=0
 neutral_offsets=offsets[None,None].expand(1,2,22,3).to('cuda')
 bones=(torch.arange(1,22),torch.tensor(PARENTS[1:]))
 results=[]
 indices=np.linspace(0,len(requests)-1,min(args.limit,len(requests)),dtype=int)
 for filename in ('best.pth','latest.pth'):
  path=control/filename;saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
  assert set(saved['trainable_state'])==names
  missing=model.load_state_dict(saved['trainable_state'],strict=False)
  assert not missing.unexpected_keys and not (set(missing.missing_keys)&names)
  mean=saved['mean'].to('cuda');std=saved['std'].to('cuda');step=saved['step']
  assert torch.equal(mean[:262].cpu(),torch.from_numpy(data.adapter.mean))
  assert torch.equal(std[:262].cpu(),torch.from_numpy(data.adapter.std))
  checkpoint={'step':step,'path':str(path),'sha256':digest(path),'cases':[]}
  for i in indices:
   request=requests.requests[int(i)]
   _,caption,a,b,length=requests[int(i)]
   seed=20261017+int(i);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   with torch.inference_mode():
    raw=model.forward_test({'text':[caption],'motion_lens':torch.tensor([length],device='cuda')})['output']
    raw=raw.reshape(1,length,2,268).float()*std+mean
    direct=raw[...,:66].reshape(1,length,2,22,3)
    fk=fk_positions(raw,neutral_offsets)
    error=(direct-fk).norm(dim=-1)[...,1:]*1000
    direct_bones=(direct[...,bones[0],:]-direct[...,bones[1],:]).norm(dim=-1)
    neutral_lengths=offsets[1:].norm(dim=-1).to('cuda')
    # Reference motion is only used for teacher-forced diagnostics, never generation.
    target_raw,target_offsets=data.pair(request['identifier'],int(request['crop_start']),bool(request['left_right_swap']),int(request['anchor_person']),bool(request['output_person_swap']))
    assert len(target_raw)==length
    expected=torch.stack((a,b),1)[:length].numpy()*data.adapter.std+data.adapter.mean
    assert np.allclose(target_raw[...,:262],expected,atol=1e-5)
    target=torch.from_numpy(target_raw)[None].to('cuda');target_norm=((target-mean)/std).flatten(2)
    gt_positions=target[...,:66].reshape_as(direct)
    cond=model.text_process({'text':[caption]})['cond']
    denoise=[]
    mask=torch.ones(1,length,2,device='cuda')
    for timestep in (10,500,900):
     torch.manual_seed(seed+1000+timestep)
     ts=torch.tensor([timestep],device='cuda')
     noisy=model.decoder.diffusion.q_sample(target_norm,ts)
     pred=model.decoder.net(noisy,ts,mask=mask,cond=cond).reshape_as(target)
     pred_raw=pred*std+mean
     predicted_fk=fk_positions(pred_raw,torch.from_numpy(target_offsets)[None].to('cuda'))
     direct_gt=(pred_raw[...,:66].reshape_as(direct)-gt_positions).norm(dim=-1)*1000
     fk_gt=(predicted_fk-gt_positions).norm(dim=-1)*1000
     denoise.append({'timestep':timestep,'direct_position_mpjpe_mm':float(direct_gt.mean()),'fk_position_mpjpe_mm':float(fk_gt.mean()),'normalized_position_mse':float((pred[...,:66]-((target-mean)/std)[...,:66]).square().mean())})
    assert torch.isfinite(raw).all() and torch.isfinite(fk).all()
    file=out/f'step{step}_request{int(i)}.npz'
    np.savez(file,generated268=raw[0].cpu().numpy(),fk_mean_shape=fk[0].cpu().numpy())
    row={'request_index':int(i),'identifier':request['identifier'],'frames':length,'seed':seed,
         'direct_vs_fk_nonroot_mean_mm':float(error.mean()),'direct_vs_fk_nonroot_p95_mm':float(torch.quantile(error,.95)),
         'direct_bone_neutral_relative_abs_error':float(((direct_bones-neutral_lengths).abs()/neutral_lengths).mean()),
         'teacher_forced_denoising':denoise,'generated_array_path_cloud_only':str(file)}
    checkpoint['cases'].append(row)
    print(json.dumps({'step':step,**row}),flush=True)
  results.append(checkpoint)
  (out/'STATUS.json').write_text(json.dumps({'status':'checkpoint_diagnosed','step':step,'pid':os.getpid()})+'\n')
 report={'schema':'intergen_output_heads_diagnostic_v1','scope':'diagnostic only, not FID/SOTA; generation consumes text/length only; neutral SMPL shape for generated FK; GT shape used only for separate teacher-forced error measurement','source_sha256':digest(Path(__file__)),'results':results,'cuda':torch.cuda.get_device_name(0)}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'STATUS.json').write_text(json.dumps({'status':'complete','pid':os.getpid(),'checkpoints':[x['step'] for x in results]})+'\n')
if __name__=='__main__':main()
