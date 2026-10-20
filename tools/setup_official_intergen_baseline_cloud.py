"""Prepare an isolated official InterGen checkout/runtime without GPU work."""
from pathlib import Path
import hashlib,io,json,os,subprocess,tarfile,time
ROOT=Path('/workspace/HHC')
CONTROL=ROOT/'output/intergen_official_baseline_setup_20261017'
SOURCE=ROOT/'third_party/intergen_official_baseline_20261017'
BASE=ROOT/'third_party/intergen_base_20261008'
ENV=Path('/workspace/intergen-official-env-py38')
COMMIT='b1750df4a859443b8a69329f759ce3a7503ae179'
def write(name,x):
 p=CONTROL/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def phase(name):
 write('STATUS.json',{'status':name,'pid':os.getpid(),'time':time.time()});print(name,flush=True)
def run(args):
 print(json.dumps({'command':args}),flush=True);subprocess.run(args,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 CONTROL.mkdir(exist_ok=False)
 phase('preparing_official_source')
 assert subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip()==COMMIT
 assert not SOURCE.exists() and not ENV.exists()
 blob=subprocess.check_output(['git','-C',str(BASE),'archive','--format=tar',COMMIT])
 SOURCE.mkdir()
 with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
  for member in archive.getmembers():
   assert (SOURCE/member.name).resolve().is_relative_to(SOURCE.resolve())
   assert not member.issym() and not member.islnk()
  archive.extractall(SOURCE,filter='data')
 files={str(p.relative_to(SOURCE)):digest(p) for p in SOURCE.rglob('*') if p.is_file()}
 # Paths and checkpoint cadence only. Keep stock batch32, FP32 CLI, all model/loss/optimizer/scheduler code.
 train=SOURCE/'configs/train.yaml';original=train.read_text()
 edited=original.replace('EXP_NAME: IG-S-8','EXP_NAME: official_intergen_scratch_20261017').replace('CHECKPOINT: ./checkpoints','CHECKPOINT: /workspace/HHC/output/intergen_official_baseline_20261017').replace('SAVE_EPOCH: 100','SAVE_EPOCH: 1')
 assert edited!=original
 train.write_text(edited)
 data_root=SOURCE/'data/interhuman_processed';data_root.mkdir(parents=True,exist_ok=True)
 actual=ROOT/'dataset/interhuman_official'
 links={key:actual/key for key in ('motions_processed','annots')}
 links.update({key+'.txt':actual/'split'/(key+'.txt') for key in ('train','val','test')})
 for name,target in links.items():
  assert target.exists(),target
  (data_root/name).symlink_to(target,target_is_directory=target.is_dir())
 ignore=actual/'ignore_list.txt'
 if ignore.exists():(data_root/'ignore_list.txt').symlink_to(ignore)
 stats={}
 for suffix in ('mean','std'):
  p=SOURCE/'data'/('global_'+suffix+'.npy');assert p.exists()
  stats[suffix]={'sha256':digest(p),'bytes':p.stat().st_size}
 mismatches={name:{'before':expected,'after':digest(SOURCE/name)} for name,expected in files.items() if digest(SOURCE/name)!=expected}
 assert set(mismatches)=={'configs/train.yaml'},mismatches
 write('SOURCE_PROVENANCE.json',{'commit':COMMIT,'archive_sha256':hashlib.sha256(blob).hexdigest(),'tracked_files':files,'modified':mismatches,'training_config_changes':['output paths/name','checkpoint every epoch instead of every100 epochs'],'data_links':{k:str(v) for k,v in links.items()},'ignore_list_present':ignore.exists(),'stats':stats,'no_training_code_changes':True})
 phase('installing_python38')
 run(['/usr/bin/uv','python','install','3.8'])
 run(['/usr/bin/uv','venv','--python','3.8',str(ENV)])
 phase('installing_official_dependencies')
 requirements=CONTROL/'requirements.lock-input.txt'
 requirements.write_text('\n'.join(['torch==1.13.1+cu117','torchvision==0.14.1+cu117','torchaudio==0.13.1+cu117','lightning==1.9.1','pytorch-lightning==1.9.1','torchmetrics==0.11.4','numpy==1.23.5','scipy==1.4.1','matplotlib==3.2.0','Pillow==7.2.0','yacs==0.1.8','tqdm==4.65.0','tensorboard==2.13.0','setuptools==68.2.2','ftfy==6.1.3','regex==2023.12.25','packaging==23.2','git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'])+'\n')
 run(['/usr/bin/uv','pip','install','--python',str(ENV/'bin/python'),'--extra-index-url','https://download.pytorch.org/whl/cu117','--index-strategy','unsafe-best-match','-r',str(requirements)])
 phase('cpu_import_preflight')
 code="import os,sys,json;os.chdir("+repr(str(SOURCE))+");sys.path.insert(0,os.getcwd());import torch,lightning.pytorch as pl,clip;from tools.train import LitTrainModel;from configs import get_config;print(json.dumps({'torch':torch.__version__,'lightning':pl.__version__,'cuda_initialized':torch.cuda.is_initialized(),'entry':__import__('inspect').getfile(LitTrainModel),'model_input':get_config('configs/model.yaml').INPUT_DIM,'train_config':str(get_config('configs/train.yaml'))}));assert not torch.cuda.is_initialized()"
 run([str(ENV/'bin/python'),'-c',code])
 result=subprocess.check_output(['/usr/bin/uv','pip','freeze','--python',str(ENV/'bin/python')],text=True)
 (CONTROL/'environment.freeze.txt').write_text(result)
 phase('ready_for_official_gpu_training')
 print(json.dumps({'source':str(SOURCE),'python':str(ENV/'bin/python'),'command':[str(ENV/'bin/python'),'-u','tools/train.py'],'cwd':str(SOURCE)}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  if CONTROL.exists():write('STATUS.json',{'status':'setup_failed','pid':os.getpid(),'error':repr(error),'time':time.time()})
  raise
