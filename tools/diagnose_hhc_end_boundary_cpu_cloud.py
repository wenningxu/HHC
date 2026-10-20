"""CPU-only END-context diagnostic; never changes a queued sampler or scores FID."""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
OUT=ROOT/'output/hhc_end_token_boundary_cpu_20261017'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');os.replace(t,p)

def main():
 assert str(ROOT)=='/workspace/HHC' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 os.nice(15);OUT.mkdir(exist_ok=False)
 import torch,numpy as np,clip
 import torch.nn.functional as F
 from types import SimpleNamespace
 from models.mgt_interhuman_smpl import SMPLMGT
 from models.mgt_interhuman_smpl_cfg import configure_conditioning
 from utils.interhuman_smpl_mgt import TokenData
 from utils.hhc_frozen_vq import load_frozen_vq
 from utils.hhc_cpu_text_cache import encode_text_cpu_float32
 from utils.runtime import generate_src_mask
 from utils.motion_masks import sample_historical_mmm_corruption_with_kinematic_ratios
 from pause_hhc_for_intergen_cloud import sources,read
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 policy=read(ROOT/'migration/hhc-compute-priority.json');bound=sources(policy)
 write('STATUS.json',{'status':'loading_cpu_metadata','pid':os.getpid(),'time_unix':time.time()})
 cache_path=ROOT/'output/hhc_cfg_A_historical_40k_20261016/tokens.pt'
 assert sha(cache_path)=='9c31d30b17354a20f675b0123846fadf413aa2830814197b3bfb4a0b0a6bfdcd'
 cache=torch.load(cache_path,map_location='cpu',weights_only=False,mmap=True);rows=cache['rows'];data=TokenData(rows)
 counts=[len(row['tokens']) for row in rows]
 assert len(counts)==4567 and sum(n>=75 for n in counts)==215 and sum(n>75 for n in counts)==210
 histogram={str(n):counts.count(n) for n in sorted(set(counts))}
 selected=[i for i,row in enumerate(rows) if len(row['tokens'])>75]
 indices=[selected[i] for i in (0,len(selected)//2,len(selected)-1)]
 # Exercise the actual sampler method with fake logits solely to trace its inputs.
 traces=[]
 for length in (1,74,75):
  seen=[]
  def forward(ids,*args,**kwargs):
   seen.append({'input_tokens':ids.shape[1],'end_count':int((ids==7).sum()),'valid_attention':int(args[1].sum())})
   return torch.zeros(*ids.shape,7)
  fake=SimpleNamespace(num_vq=7,block_size=76,forward_function=forward)
  output=SMPLMGT.sample_intra_cas(fake,torch.zeros(1,512),torch.zeros(1,77,512),torch.zeros(1,20),torch.ones(1,2,dtype=torch.long),torch.tensor([length*4]),[114514],max_steps=1,temperature=0,CFG=0)
  assert output.shape==(1,75,2)
  assert seen[0]['end_count']==(0 if length==75 else 2)
  traces.append({'native_token_length':length,'first_forward':seen[0],'scope':'fake logits only, production sampler input trace; not a quality result'})
 write('STATUS.json',{'status':'loading_saved_A4000_cpu_weights','pid':os.getpid(),'time_unix':time.time()})
 checkpoint=ROOT/'output/hhc_after_official_intergen_resume_control_20261017/saved_A_step4000.pth'
 assert checkpoint.stat().st_size==3429290971 and sha(checkpoint)=='fb1b5a2c606ffaa68fcff878f1038d122713d652b1767de35c64acd0f8614963'
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
 assert saved['step']==4000 and saved['contract_sha256']==policy['hhc_contract_sha256']
 vq,_,_,spec=load_frozen_vq(policy['vq_spec'],'cpu',ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004/manifest.json')
 net=configure_conditioning(SMPLMGT(vq),'historical',sampling=True).eval()
 result=net.load_state_dict(saved['model'],strict=False)
 assert not result.unexpected_keys and all(k.startswith('trans_base.vqvae.') for k in result.missing_keys)
 net.requires_grad_(False);vq.eval()
 url=clip.clip._MODELS['ViT-B/32'];cached=Path.home()/'.cache/clip'/url.rsplit('/',1)[1]
 assert cached.is_file() and sha(cached)==url.split('/')[-2]
 text_model,_=clip.load('ViT-B/32',device='cpu',jit=False);text_model.eval().requires_grad_(False)
 samples=[data[(index,20261017)] for index in indices]
 assert all(n==75 and target.shape==(76,2) and (target[75]==12288).all() for target,n,text,beta in samples)
 with torch.no_grad():global_text,word=encode_text_cpu_float32(text_model,[item[2] for item in samples])
 del text_model
 report={'status':'running','checkpoint_step':4000,'checkpoint_sha256':sha(checkpoint),'vq_sha256':spec['sha256'],'token_cache_sha256':sha(cache_path),'training_cache_rows':len(rows),'train_token_length_hist':histogram,'train_n75_without_trim':sum(n>=75 for n in counts),'train_n75_after_drop_one':sum(n>75 for n in counts),'sampler_shape_trace':traces,'precision':'CPU FP32, not GPU BF16; teacher-forced GT tokens/shapes for diagnostics only','sampling_cfg':3.5,'results':[],'sources':{p:sha(ROOT/p) for p in ['tools/diagnose_hhc_end_boundary_cpu_cloud.py','models/mgt_interhuman_smpl.py','models/mgt_interhuman_smpl_cfg.py','models/two_person_transformer.py','models/motion_transformer.py','models/trans_encdec.py','utils/interhuman_smpl_mgt.py']},'no_fid_or_generation_quality_claim':True,'changed_training_or_sampler_source':False}
 def stats(short,full,target,support):
  short,full=short.float(),full.float();assert torch.isfinite(short).all() and torch.isfinite(full).all()
  delta=(short-full).abs();ls=short.log_softmax(-1);lf=full.log_softmax(-1)
  nll_s=F.cross_entropy(short[support],target[support]);nll_f=F.cross_entropy(full[support],target[support])
  return {'max_abs_logit_difference':float(delta.max()),'mean_abs_logit_difference':float(delta.mean()),'top1_agreement':float((short.argmax(-1)==full.argmax(-1)).float().mean()),'mean_KL_full_to_short_nats':float((lf.exp()*(lf-ls)).sum(-1).mean()),'support_tokens':int(support.sum()),'teacher_forced_nll_current75':float(nll_s),'teacher_forced_nll_full76':float(nll_f),'nll_full_minus_current':float(nll_f-nll_s),'allclose_rtol1e_5_atol2e_4':bool(torch.allclose(short,full,rtol=1e-5,atol=2e-4))}
 with torch.no_grad():
  for sample_index,(target,_,caption,beta) in enumerate(samples):
   for length in (74,75):
    for pattern in ('all_mask','historical_corruption'):
     ground=torch.full((1,76,2),12289,dtype=torch.long);ground[0,:length]=target[:length];ground[0,length]=12288
     valid=generate_src_mask(76,torch.tensor([length]))[:,:,None].expand_as(ground)
     if pattern=='all_mask':inputs=ground.masked_fill(valid,12290);support=valid
     else:
      inputs,support,_,_=sample_historical_mmm_corruption_with_kinematic_ratios(ground,valid,12288,pkeep=.5,generator=torch.Generator().manual_seed(20261017+sample_index*100+length))
     conditional=[];unconditional=[]
     for width in (75,76):
      args=(inputs[:,:width],global_text[sample_index:sample_index+1],generate_src_mask(width,torch.tensor([length+1])))
      kwargs={'word_emb':word[sample_index:sample_index+1],'betas':beta[None],'gender':torch.ones(1,2,dtype=torch.long)}
      conditional.append(net.forward_function(*args,**kwargs)[:,:length]);unconditional.append(net.forward_function(*args,**kwargs,force_mask=True)[:,:length])
     cfg=[u+3.5*(c-u) for c,u in zip(conditional,unconditional)]
     row={'name':rows[indices[sample_index]]['name'],'native_token_length':length,'case':'masked_PAD_extension_control' if length==74 else 'actual_missing_END_context_boundary','corruption':pattern,'conditioned':stats(*conditional,ground[:,:length],support[:,:length]),'cfg3_5':stats(*cfg,ground[:,:length],support[:,:length])}
     report['results'].append(row);write('REPORT.partial.json',report)
     write('STATUS.json',{'status':'probing_cpu_forward_only','pid':os.getpid(),'completed_cases':len(report['results']),'last_name':row['name'],'time_unix':time.time()});print(json.dumps(row),flush=True)
 assert not torch.cuda.is_initialized() and sources(policy)==bound
 report.update(status='complete_cpu_diagnostic_not_quality_validation',cpu_cuda_initialized=torch.cuda.is_initialized(),time_unix=time.time())
 write('REPORT.json',report);write('STATUS.json',{'status':report['status'],'pid':os.getpid(),'completed_cases':len(report['results']),'report_sha256':sha(OUT/'REPORT.json'),'time_unix':time.time()})
 print(json.dumps({'status':report['status'],'cases':len(report['results']),'report_sha256':sha(OUT/'REPORT.json')}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if OUT.exists():write('STATUS.json',{'status':'failed_cpu_diagnostic','pid':os.getpid(),'error':repr(error),'time_unix':time.time()})
  raise
