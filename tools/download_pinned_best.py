"""Collect the immutable first validated scratch best without touching training."""
import msvcrt,os,time
from pathlib import Path
from local_compute_control import download,event,atomic_json
ROOT=Path(__file__).resolve().parents[1]
def main():
 control=ROOT/'migration/control'
 with (control/'best010000.lock').open('a+b') as lock:
  lock.seek(0)
  if not lock.read(1):lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  atomic_json(control/'best010000.pid.json',{'pid':os.getpid(),'started':time.time()})
  while True:
   try:
    download('/dev/shm/hhc_local_control_20261015/output/intergen_rotation_scratch_20261010/best_transfer_step010000.pth',
             ROOT/'checkpoints/local_backups/intergen_scratch_20261010_step010000/best.pth',
             'e432c7a2ed82ae3fbfbb7bcdb00525e241eee0e44d35297a9b23e66d0454c2dd',729644528)
    return
   except Exception as exc:
    event('best010000_download_retry',error=str(exc));time.sleep(60)
if __name__=='__main__':main()
