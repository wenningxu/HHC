"""Cloud CPU probe of the untouched official eval dataset/model/evaluator."""
import argparse,hashlib,json,os,sys,gc,time
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and str(args.source).startswith('/workspace/HHC/third_party/')
 assert not args.output.exists()
 os.nice(10);os.chdir(args.source);sys.path.insert(0,str(args.source))
 import torch,numpy as np,clip
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 # Prevent the official constructors from initiating a new model download.
 url=clip.clip._MODELS['ViT-L/14@336px'];cached=Path.home()/'.cache/clip'/url.rsplit('/',1)[1]
 assert cached.is_file() and digest(cached)==url.split('/')[-2]
 from configs import get_config
 from datasets import get_dataset_motion_loader
 from models import InterGen
 from datasets.evaluator import EvaluatorModelWrapper
 loader,data=get_dataset_motion_loader(get_config('configs/datasets.yaml').interhuman_test,96)
 ids=sorted(row['name'] for row in data.data_list)
 assert len(data)==1098 and len(loader)==11 and len(set(ids))==1098
 sample=next(iter(loader));assert sample[2].shape==sample[3].shape==(96,300,262)
 assert torch.isfinite(sample[2]).all() and torch.isfinite(sample[3]).all()
 assert int(sample[4].min())>=14 and int(sample[4].max())<=299
 report={'status':'cpu_preflight_passed_no_gpu_execution','source':str(args.source),'test_dataset_rows':len(data),'main_batches':len(loader),'main_samples_per_rep':len(loader)*96,'ids':ids,'first_batch_shapes':[list(x.shape) for x in (sample[2],sample[3])],'first_batch_length_min':int(sample[4].min()),'first_batch_length_max':int(sample[4].max()),'test_ids_sha256':hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()}
 del loader,data,sample;gc.collect()
 model=InterGen(get_config('configs/model.yaml'))
 saved=torch.load(args.checkpoint,map_location='cpu');state=saved['state_dict']
 for k in list(state.keys()):
  if 'model' in k:state[k.replace('model.','')]=state.pop(k)
 model.load_state_dict(state,strict=True)
 assert all(torch.isfinite(v).all() for v in model.state_dict().values())
 report.update(probe_checkpoint=str(args.checkpoint),probe_checkpoint_sha256=digest(args.checkpoint),probe_step=int(saved['global_step']),strict_model_state_keys=len(model.state_dict()))
 del model,state,saved;gc.collect()
 evaluator=EvaluatorModelWrapper(get_config('configs/eval_model.yaml'),torch.device('cpu'))
 assert all(torch.isfinite(v).all() for v in evaluator.model.state_dict().values())
 report.update(evaluator_checkpoint_sha256=digest(Path('eval_model/interclip.ckpt')),strict_evaluator_state_keys=len(evaluator.model.state_dict()),torch_version=torch.__version__,cpu_cuda_initialized=torch.cuda.is_initialized(),time_unix=time.time(),source_sha256=digest(Path(__file__)))
 assert report['evaluator_checkpoint_sha256']=='4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8' and not report['cpu_cuda_initialized']
 args.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='ids'}),flush=True)
if __name__=='__main__':main()
