"""Stage pinned public assets on the attributed Runpod compute checkout."""
import hashlib,json,os,pathlib,time,zipfile
import requests
ROOT=pathlib.Path('/workspace/HHC')
def fetch(url,target,size,sha,params=None):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        with target.open('rb') as f: assert target.stat().st_size==size and hashlib.file_digest(f,'sha256').hexdigest()==sha
        return
    partial=target.with_suffix(target.suffix+'.partial')
    for attempt in range(30):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset<size:
                with requests.get(url,params=params,headers={'Range':f'bytes={offset}-'},stream=True,timeout=(30,90)) as r:
                    r.raise_for_status()
                    if offset: assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    with partial.open('ab') as out:
                        for block in r.iter_content(1024*1024):out.write(block)
            with partial.open('rb') as f: assert partial.stat().st_size==size and hashlib.file_digest(f,'sha256').hexdigest()==sha
            os.replace(partial,target)
            print(json.dumps({'verified':str(target),'bytes':size,'sha256':sha}),flush=True)
            return
        except Exception as exc:
            print(json.dumps({'retry':attempt,'file':str(target),'error':str(exc)}),flush=True);time.sleep(15)
    raise RuntimeError(f'Asset staging failed: {target}')
url='https://drive.usercontent.google.com/download'
fetch(url,ROOT/'dataset/interhuman_official/motions_processed.zip',2462028219,'ad951d24f95fff550d964c8f977eb99da47264b81e3080312c95046e69fd6325',{'id':'1qpFij33tVZhUkk9-kDaDy1FuQb0LJSEf','export':'download','confirm':'t'})
base=ROOT/'dataset/interhuman_official'
fast=pathlib.Path('/dev/shm/hhc_inputs/dataset/interhuman_official');fast.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(base/'motions_processed.zip') as z:
    for info in z.infolist():
        destination=(fast/info.filename).resolve()
        assert destination.is_relative_to(fast.resolve()) and info.filename.startswith('motions_processed/')
        z.extract(info,fast)
canonical=base/'motions_processed'
if not canonical.is_symlink():
    if canonical.exists():canonical.rename(base/f'motions_processed.partial_persistent_{time.time_ns()}')
    canonical.symlink_to(fast/'motions_processed',target_is_directory=True)
fetch(url,ROOT/'third_party/InterMask/checkpoints/eval_model/interclip.ckpt',1513626777,'4e14b8ee5c2d965778fe003fb37d98ed8e5c7e0940668515fc7dcfa2831c17f8',{'id':'1bJv5lTP7otJleaBYZ2byjru_k_wCsGvH','export':'download','confirm':'t'})
from clip import clip
for name in ['ViT-B/32','ViT-L/14@336px']:
    path=clip._download(clip._MODELS[name],str(pathlib.Path.home()/'.cache/clip'))
    print(json.dumps({'clip_verified':name,'file':path,'sha256':hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()}),flush=True)
(ROOT/'ASSETS_READY.json').write_text(json.dumps({'status':'verified_public_assets','time':time.time()}))
