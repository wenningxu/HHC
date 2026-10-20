"""Collect the additional pinned frozen training CLIP independently of compute."""
import hashlib,json,msvcrt,os,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'migration/download_packages'))
import requests
from local_compute_control import atomic_json,event
SHA='3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'
SIZE=934088680
URL=f'https://openaipublic.azureedge.net/clip/models/{SHA}/ViT-L-14-336px.pt'
CONTROL=ROOT/'migration/control'
def main():
    with (CONTROL/'training-clip.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        atomic_json(CONTROL/'training-clip.pid.json',{'pid':os.getpid(),'started':time.time()})
        os.environ.setdefault('HTTPS_PROXY','http://127.0.0.1:7892')
        target=ROOT/'runtime_assets/clip/ViT-L-14-336px.pt';target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_suffix('.pt.partial')
        while not target.exists():
            try:
                start=partial.stat().st_size if partial.exists() else 0
                assert start<=SIZE
                if start<SIZE:
                    with requests.get(URL,headers={'Range':f'bytes={start}-'},stream=True,timeout=(20,90)) as r:
                        r.raise_for_status()
                        assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{SIZE-1}/{SIZE}'
                        progress=start
                        with partial.open('ab') as out:
                            for block in r.iter_content(1024*1024):
                                assert out.tell()+len(block)<=SIZE
                                out.write(block)
                                if out.tell()-progress>=64*1024*1024:
                                    progress=out.tell();atomic_json(CONTROL/'training-clip-status.json',{'status':'downloading','bytes':progress,'total':SIZE,'sha256':SHA})
                with partial.open('rb') as source:assert partial.stat().st_size==SIZE and hashlib.file_digest(source,'sha256').hexdigest()==SHA
                os.replace(partial,target)
            except Exception as exc:event('training_clip_retry',error=str(exc));time.sleep(3600)
        with target.open('rb') as source:assert target.stat().st_size==SIZE and hashlib.file_digest(source,'sha256').hexdigest()==SHA
        receipt={'file':str(target),'bytes':SIZE,'sha256':SHA,'source_url':URL,'verified_at':time.time(),'cloud_status':'pending'}
        atomic_json(target.with_suffix('.pt.receipt.json'),receipt)
        atomic_json(CONTROL/'training-clip-status.json',dict(receipt,status='complete'))
        event('training_clip_verified',**receipt)
if __name__=='__main__':main()
