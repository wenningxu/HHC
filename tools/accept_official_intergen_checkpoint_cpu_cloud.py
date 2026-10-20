"""Verify a persistent cloud snapshot of an official Lightning checkpoint, CPU only."""
from pathlib import Path
import argparse,hashlib,json,os,time
ROOT=Path('/workspace/HHC');OUT=ROOT/'output/intergen_official_baseline_20261017'
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def active(launch):
 p=Path('/proc')/str(launch['pid'])
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  return fields[19]==launch['birth'] and fields[0] not in ('Z','X')
 except FileNotFoundError:return False

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 launch=json.loads((OUT/'LAUNCH.json').read_text());source=Path(launch['cwd'])
 manifest=json.loads((ROOT/'output/intergen_official_baseline_setup_20261017/SOURCE_PROVENANCE.json').read_text())
 for name,expected in manifest['tracked_files'].items():
  expected=launch.get('source_changes',{}).get(name,{}).get('after',expected)
  assert digest(source/name)==expected,name
 assert digest(ROOT/'tools/run_official_intergen_checkpointed_cloud.py')==launch['runtime_changes']['wrapper_sha256']
 files=list((OUT/'official_intergen_scratch_20261017/model').glob('*.ckpt'))
 if not files:print(json.dumps({'status':'not_ready','reason':'no_checkpoint_yet'}));raise SystemExit(2)
 original=max(files,key=lambda p:p.stat().st_mtime_ns);stat=original.stat()
 assert stat.st_mtime>=launch['time'] and time.time()-stat.st_mtime>=5,'Checkpoint must be stable and belong to this launch'
 folder=OUT/'accepted_checkpoints';folder.mkdir(exist_ok=True);snapshot=folder/original.name
 if not snapshot.exists():os.link(original,snapshot)
 assert snapshot.stat().st_ino==stat.st_ino and snapshot.stat().st_size==stat.st_size
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 saved=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
 step=int(saved['global_step']);assert step>=352 and step%352==0
 state=saved['state_dict']
 assert tuple(state['model.decoder.net.motion_embed.weight'].shape)==(1024,262)
 assert tuple(state['model.decoder.net.out.linear.weight'].shape)==(262,1024)
 assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
 assert saved['optimizer_states'] and saved['lr_schedulers']
 optim=saved['optimizer_states'][0];steps=[int(v['step']) for v in optim['state'].values()]
 assert steps and min(steps)==max(steps)==step
 assert all(torch.isfinite(v).all() for item in optim['state'].values() for v in item.values() if isinstance(v,torch.Tensor))
 scheduler=saved['lr_schedulers'][0];assert scheduler['warmup']==10 and scheduler['max_num_iters']==2000
 if args.final:
  assert step==704000 and not active(launch)
  log=Path(launch['log_path']).read_text();assert 'max_epochs=2000' in log and 'reached' in log
 receipt={'status':'verified_final' if args.final else 'verified_partial','step':step,'epoch':saved['epoch'],'original_checkpoint':str(original),'snapshot':str(snapshot),'bytes':snapshot.stat().st_size,'sha256':digest(snapshot),'optimizer_states':len(steps),'optimizer_step_min':min(steps),'optimizer_step_max':max(steps),'scheduler':scheduler,'input_dim':262,'source_commit':launch['source_commit'],'runtime_wrapper_sha256':launch['runtime_changes']['wrapper_sha256'],'attributed_trainer_live':active(launch),'checkpoint_top_level_rng_present':any('rng' in k.lower() for k in saved),'rng_note':'Default official Lightning checkpoint does not promise RNG or dataloader cursor restoration; no bitwise resume claim','cpu_only':True,'cpu_cuda_initialized':torch.cuda.is_initialized(),'time_unix':time.time(),'verifier_source_sha256':digest(Path(__file__))}
 assert not torch.cuda.is_initialized()
 p=OUT/('FINAL_CHECKPOINT_ACCEPTANCE.json' if args.final else 'STARTUP_CHECKPOINT_ACCEPTANCE.json');p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
