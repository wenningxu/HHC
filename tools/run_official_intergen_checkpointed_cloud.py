"""Run unchanged official training with nonreentrant activation recomputation only."""
from pathlib import Path
import argparse,hashlib,json,os,runpy,sys
ROOT=Path('/workspace/HHC');SOURCE=ROOT/'third_party/intergen_official_baseline_20261017'
sys.path.insert(0,str(SOURCE));os.chdir(SOURCE)
import torch
from torch.utils.checkpoint import checkpoint


def install_checkpointing():
 from models.blocks import TransformerBlock
 original=TransformerBlock.forward
 def recomputed(self,*args,**kwargs):
  if self.training and torch.is_grad_enabled():
   return checkpoint(lambda *inner:original(self,*inner,**kwargs),*args,use_reentrant=False,preserve_rng_state=True)
  return original(self,*args,**kwargs)
 TransformerBlock.forward=recomputed
 return original


def probe():
 from models.nets import InterDenoiser
 torch.set_num_threads(2);torch.set_float32_matmul_precision('medium');torch.manual_seed(20261017)
 model=InterDenoiser(262,1024,ff_size=2048,num_layers=8,num_heads=8,dropout=.1).cuda().train()
 # The official zero output head would hide interior gradient errors; engage all blocks.
 torch.nn.init.normal_(model.out.linear.weight,mean=0,std=.01)
 x=torch.randn(2,17,524,device='cuda',requires_grad=True)
 cond=torch.randn(2,768,device='cuda',requires_grad=True)
 mask=torch.ones(2,17,2,device='cuda');mask[1,12:]=0
 t=torch.tensor([10,900],device='cuda');target=torch.randn_like(x)
 cpu_state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state_all()
 out=model(x,t,mask=mask,cond=cond);loss=(out-target).square().mean();loss.backward()
 reference={'output':out.detach().cpu(),'x_grad':x.grad.detach().cpu(),'cond_grad':cond.grad.detach().cpu(),'grads':{n:v.grad.detach().cpu() if v.grad is not None else None for n,v in model.named_parameters()},'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
 model.zero_grad(set_to_none=True);x.grad=None;cond.grad=None;del out,loss
 torch.cuda.empty_cache();install_checkpointing();torch.set_rng_state(cpu_state);torch.cuda.set_rng_state_all(cuda_state)
 out=model(x,t,mask=mask,cond=cond);loss=(out-target).square().mean();loss.backward()
 output_diff=float((out.detach().cpu()-reference['output']).abs().max());gradient_max=0.;compared=0
 for n,v in model.named_parameters():
  previous=reference['grads'][n]
  if previous is None:assert v.grad is None;continue
  assert v.grad is not None and torch.isfinite(v.grad).all();gradient_max=max(gradient_max,float((v.grad.detach().cpu()-previous).abs().max()));compared+=1
 input_diff=float((x.grad.detach().cpu()-reference['x_grad']).abs().max())
 cond_diff=float((cond.grad.detach().cpu()-reference['cond_grad']).abs().max())
 rng_equal=torch.equal(torch.get_rng_state(),reference['cpu_rng']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),reference['cuda_rng']))
 assert output_diff<1e-6 and gradient_max<1e-5 and input_diff<1e-6 and cond_diff<1e-6 and rng_equal
 result={'status':'passed','scope':'single-process full8-layer official InterDenoiser, FP32, dropout0.1, both people, padded mask; nonzero output head to exercise interior gradients; not bitwise multi-step training proof','output_max_abs':output_diff,'parameter_gradient_max_abs':gradient_max,'input_gradient_max_abs':input_diff,'condition_gradient_max_abs':cond_diff,'parameter_gradients_compared':compared,'cpu_cuda_rng_equal_after_backward':rng_equal,'runtime_wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch':torch.__version__,'cuda':torch.cuda.get_device_name(0),'max_memory_allocated':torch.cuda.max_memory_allocated()}
 path=ROOT/'output/intergen_official_baseline_20261017/ACTIVATION_CHECKPOINT_PROBE.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');args=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and str(Path(__file__).resolve().parents[1])==str(ROOT)
 if args.probe:return probe()
 report=json.loads((ROOT/'output/intergen_official_baseline_20261017/ACTIVATION_CHECKPOINT_PROBE.json').read_text())
 assert report['status']=='passed' and report['runtime_wrapper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 install_checkpointing()
 # Every training/data/loss/optimizer/scheduler function still comes from official entry.
 sys.argv=[str(SOURCE/'tools/train.py')]
 runpy.run_path(str(SOURCE/'tools/train.py'),run_name='__main__')
if __name__=='__main__':main()
