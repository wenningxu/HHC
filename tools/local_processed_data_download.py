"""Resume the official processed archive and verify pinned SHA256 and ZIP CRCs."""
import hashlib,json,msvcrt,os,pathlib,shutil,sys,time,zipfile,zlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'migration/download_packages'))
import requests
from local_compute_control import atomic_json,event
SIZE=2462028219
SHA='ad951d24f95fff550d964c8f977eb99da47264b81e3080312c95046e69fd6325'
def main():
 control=ROOT/'migration/control'
 with (control/'processed-data.lock').open('a+b') as lock:
  lock.seek(0)
  if not lock.read(1):lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  atomic_json(control/'processed-data.pid.json',{'pid':os.getpid(),'started':time.time()})
  os.environ.setdefault('HTTPS_PROXY','http://127.0.0.1:7892')
  base=ROOT/'dataset/interhuman_official';target=base/'motions_processed.zip';partial=target.with_suffix('.zip.partial')
  state=control/'processed-data-status.json'
  if not target.exists():
   while not partial.exists() or partial.stat().st_size<SIZE:
    start=partial.stat().st_size if partial.exists() else 0;end=min(start+64*1024*1024,SIZE)-1
    try:
     with requests.get('https://drive.usercontent.google.com/download',params={'id':'1qpFij33tVZhUkk9-kDaDy1FuQb0LJSEf','export':'download','confirm':'t'},headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
      r.raise_for_status()
      assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{SIZE}', 'Unexpected range response'
      with partial.open('ab') as f:
       for block in r.iter_content(1024*1024):f.write(block)
     assert partial.stat().st_size==end+1
     atomic_json(state,{'status':'downloading','bytes':end+1,'total':SIZE,'source_file_id':'1qpFij33tVZhUkk9-kDaDy1FuQb0LJSEf'})
    except Exception as exc:
     event('processed_data_retry',error=str(exc));time.sleep(60)
   assert partial.stat().st_size==SIZE and hashlib.file_digest(partial.open('rb'),'sha256').hexdigest()==SHA
   os.replace(partial,target)
  assert target.stat().st_size==SIZE and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==SHA
  count=0
  with zipfile.ZipFile(target) as z:
   assert shutil.disk_usage(base).free>sum(x.file_size for x in z.infolist())+1024**3
   for info in z.infolist():
    assert info.filename.startswith('motions_processed/')
    destination=(base/info.filename).resolve();assert destination.is_relative_to(base.resolve())
    if info.is_dir():destination.mkdir(parents=True,exist_ok=True);continue
    if destination.exists():
     assert destination.stat().st_size==info.file_size and zlib.crc32(destination.read_bytes())==info.CRC
    else:
     destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix(destination.suffix+'.part')
     with z.open(info) as src,tmp.open('wb') as out:shutil.copyfileobj(src,out)
     assert tmp.stat().st_size==info.file_size;os.replace(tmp,destination)
    count+=1
  atomic_json(state,{'status':'complete','archive_bytes':SIZE,'archive_sha256':SHA,'extracted_files':count,'source_file_id':'1qpFij33tVZhUkk9-kDaDy1FuQb0LJSEf'})
  event('processed_data_verified',files=count,bytes=SIZE,sha256=SHA)
if __name__=='__main__':main()
