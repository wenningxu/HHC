"""Isolate the original official eval20 entry without touching live train inputs."""
from pathlib import Path
import ast,hashlib,io,json,os,subprocess,tarfile,time
ROOT=Path('/workspace/HHC');BASE=ROOT/'third_party/intergen_base_20261008'
SOURCE=ROOT/'third_party/intergen_official_reproduction_20261017'
CONTROL=ROOT/'output/intergen_official_reproduction_setup_20261017'
RUN=ROOT/'output/intergen_official_baseline_20261017'
ENV=Path('/workspace/intergen-official-env-py38/bin/python')
COMMIT='b1750df4a859443b8a69329f759ce3a7503ae179'

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,path)
def phase(status,**kw):
 write(CONTROL/'STATUS.json',{'status':status,'pid':os.getpid(),'time_unix':time.time(),**kw});print(status,flush=True)

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and ROOT.is_dir()
 CONTROL.mkdir(exist_ok=False);phase('isolating_official_eval_source')
 assert not SOURCE.exists()
 assert subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip()==COMMIT
 blob=subprocess.check_output(['git','-C',str(BASE),'archive','--format=tar',COMMIT]);SOURCE.mkdir()
 with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
  for member in archive.getmembers():
   assert (SOURCE/member.name).resolve().is_relative_to(SOURCE.resolve()) and not member.issym() and not member.islnk()
  archive.extractall(SOURCE,filter='data')
 original={str(p.relative_to(SOURCE)):digest(p) for p in SOURCE.rglob('*') if p.is_file()}
 future=RUN/'reproduction/final_intergen.ckpt';cfg=SOURCE/'configs/model.yaml';text=cfg.read_text();needle='CHECKPOINT: checkpoints/intergen.ckpt'
 assert text.count(needle)==1;cfg.write_text(text.replace(needle,'CHECKPOINT: '+str(future)))
 # Real directories, per-file links: os.walk does not follow nested directory links.
 actual=ROOT/'dataset/interhuman_official';view=CONTROL/'data_view';view.mkdir()
 for person in ('person1','person2'):
  directory=view/'motions_processed'/person;directory.mkdir(parents=True)
  for p in (actual/'motions_processed'/person).glob('*.npy'):
   (directory/p.name).symlink_to(p.resolve())
 (view/'annots').symlink_to((actual/'annots').resolve(),target_is_directory=True)
 for split in ('train','val','test'):(view/(split+'.txt')).symlink_to((actual/'split'/(split+'.txt')).resolve())
 bad={}
 for name in (actual/'split/test.txt').read_text().splitlines():
  p=actual/'annots'/(name.strip()+'.txt')
  try:p.read_text(encoding='utf-8')
  except UnicodeDecodeError:bad[name.strip()]={'reason':'non_utf8','sha256':digest(p)}
 assert set(bad)=={'4642'},bad
 (view/'ignore_list.txt').write_text(''.join(name+'\n' for name in sorted(bad)))
 (SOURCE/'data/interhuman_processed').symlink_to(view,target_is_directory=True)
 evaluator=ROOT/'third_party/InterMask/checkpoints/eval_model/interclip.ckpt'
 assert evaluator.stat().st_size==1513626777 and digest(evaluator)=='4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8'
 (SOURCE/'eval_model').mkdir();(SOURCE/'eval_model/interclip.ckpt').symlink_to(evaluator)
 changes={name:{'before':before,'after':digest(SOURCE/name)} for name,before in original.items() if digest(SOURCE/name)!=before}
 assert set(changes)=={'configs/model.yaml'}
 constants={node.targets[0].id:ast.literal_eval(node.value) for node in ast.walk(ast.parse((SOURCE/'tools/eval.py').read_text())) if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and isinstance(node.value,ast.Constant)}
 protocol={'mm_num_samples':100,'mm_num_repeats':30,'mm_num_times':10,'diversity_times':300,'replication_times':20,'batch_size':96}
 assert all(constants[k]==v for k,v in protocol.items())
 provenance={'status':'source_prepared_cpu_preflight_pending','source':str(SOURCE),'source_commit':COMMIT,'tracked_files':original,'modified':changes,'no_python_changes':True,'protocol':protocol,'eval_source_sha256':digest(SOURCE/'tools/eval.py'),'evaluator_checkpoint':str(evaluator),'evaluator_checkpoint_sha256':digest(evaluator),'view':str(view),'excluded_test_annotations':bad,'ignore_list_sha256':digest(view/'ignore_list.txt'),'view_links_sha256':hashlib.sha256(json.dumps({str(p.relative_to(view)):str(p.resolve()) for p in view.rglob('*') if p.is_symlink()},sort_keys=True).encode()).hexdigest(),'final_checkpoint_link':str(future),'local_checkpoint_downloads':False,'scope':'Original official final-checkpoint reproduction audit, separate from HHC fair-test selection/comparison'}
 write(CONTROL/'SOURCE_PROVENANCE.json',provenance);phase('cpu_dataset_and_strict_load_preflight')
 probe=ROOT/'tools/preflight_official_intergen_eval_cpu_cloud.py'
 command=[str(ENV),'-u',str(probe),'--source',str(SOURCE),'--checkpoint',str(RUN/'accepted_checkpoints/epoch=1-step=704.ckpt'),'--output',str(CONTROL/'CPU_PREFLIGHT.json')]
 with (CONTROL/'cpu_preflight.log').open('x') as log:
  result=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MPLBACKEND':'Agg'},stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,'CPU preflight failed; inspect original cpu_preflight.log, never launch GPU eval'
 phase('ready_for_final_training_gated_eval20',preflight_sha256=digest(CONTROL/'CPU_PREFLIGHT.json'))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if CONTROL.exists():phase('failed_needs_inspection',error=repr(error))
  raise
