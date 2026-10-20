"""CPU observer for the literal official CLI; no automatic restart or HHC launch."""
import argparse,csv,hashlib,io,json,math,os,re,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/HHC');OUT=ROOT/'output/intergen_official_baseline_20261017'
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  try:cwd=str((p/'cwd').resolve())
  except PermissionError:cwd=None  # Separate user namespaces can deny cwd even for the same host user.
  return {'pid':pid,'state':fields[0],'birth':fields[19],'cpu_ticks':int(fields[11])+int(fields[12]),'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),'cwd':cwd}
 except FileNotFoundError:return None

def observe():
 launch=json.loads((OUT/'LAUNCH.json').read_text());identity=proc(launch['pid'])
 active=bool(identity and identity['birth']==launch['birth'] and identity['state'] not in ('Z','X'))
 path=Path(launch.get('log_path',str(OUT/'train.log')))
 with path.open('rb') as f:
  size=os.fstat(f.fileno()).st_size;f.seek(max(0,size-262144));tail=f.read().decode(errors='replace')
 logs=list(re.finditer(r'epoch:\s*(\d+)\s+niter:\s*(\d+)\s+inner_iter:\s*(\d+)\s+lr:([0-9.eE+-]+)[^\r\n]*',tail))
 last=None
 if logs:
  row=logs[-1];last={'epoch':int(row[1]),'step':int(row[2]),'inner_iter':int(row[3]),'lr':float(row[4]),'losses':{k:float(v) for k,v in re.findall(r'([A-Za-z_]+):\s*(-?[0-9]+\.[0-9]+|nan|inf)',row[0]) if k not in ('epoch','niter','inner_iter','lr')}}
  assert all(math.isfinite(v) for v in last['losses'].values()),last
 state='training_verified' if active and last else 'initializing_official_entry' if active else 'completed_pending_checkpoint_acceptance' if 'max_epochs=2000' in tail and 'reached' in tail else 'terminal_needs_audit'
 rows=subprocess.run(['nvidia-smi','--query-compute-apps=pid,name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 gpu=[{'pid':int(row[0].strip()),'name':row[1].strip(),'used_memory_mib':int(row[2].strip())} for row in csv.reader(io.StringIO(rows.stdout)) if row]
 result={'status':state,'observer_pid':os.getpid(),'observed_at_unix':time.time(),'trainer':identity,'attributed_trainer_live':active,'latest_training_log':last,'log_bytes':size,'gpu_processes':gpu,'checkpoints':[{'path':str(p),'bytes':p.stat().st_size} for p in OUT.rglob('*.ckpt')],'tail':tail[-2000:] if not last or not active else None,'exit_code':'not_available_to_nonparent_observer','no_auto_restart':True,'hhc_remains_saved_step':4000}
 dest=OUT/'OBSERVER_STATUS.json';temporary=dest.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');os.replace(temporary,dest)
 with (OUT/'observations.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 print(json.dumps(result),flush=True);return active

def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 while True:
  live=observe()
  if not args.watch or not live:return
  time.sleep(600)
if __name__=='__main__':main()
