"""Download official raw InterHuman files using the frozen MD5 inventory."""
import json,msvcrt,os,pathlib,runpy,shutil,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
CONTROL=ROOT/'migration/control'
def main():
 with (CONTROL/'public-data.lock').open('a+b') as lock:
  lock.seek(0)
  if not lock.read(1):lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  (CONTROL/'public-data.pid.json').write_text(json.dumps({'pid':os.getpid(),'started':time.time()})+'\n')
  sys.path.insert(0,str(ROOT/'migration/download_packages'))
  os.environ.setdefault('HTTPS_PROXY','http://127.0.0.1:7892')
  os.environ.setdefault('HTTP_PROXY','http://127.0.0.1:7892')
  manifest=ROOT/'output/interhuman_raw_download_local_20261015.json'
  if not manifest.exists():shutil.copyfile(ROOT/'output/interhuman_raw_download_20261003.json',manifest)
  sys.argv=['download_interhuman_raw.py','--destination',str(ROOT/'dataset/interhuman_official/motions'),
            '--manifest',str(manifest),'--reuse-manifest','--workers','6']
  runpy.run_path(str(ROOT/'tools/download_interhuman_raw.py'),run_name='__main__')
if __name__=='__main__':main()
