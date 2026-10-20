"""Stage released baseline weights directly on cloud, pinned to prior hashes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'output/interhuman_baseline_restore_20261016'
ASSETS=[
    {'name':'intermask_vq','id':'1PK82nEXehGmgeng4Sp2KaAPtfX3A9E9q','bytes':518555706,
     'sha256':'af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272',
     'path':'third_party/InterMask/checkpoints/interhuman/vq_default/model/best_fid.tar'},
    {'name':'intermask_mgt','id':'1RxXTUd8CeIcFb_xhnGH7PVEavsCl3lph','bytes':651546446,
     'sha256':'392ca8c9e42c2c23aeb97ba7b810e8fa4bdcc5f4839a5a909be2f56bf1698665',
     'path':'third_party/InterMask/checkpoints/interhuman/trans_default/model/best_fid.tar'},
    {'name':'intergen','id':'1sYOJEdbB8hDI5pY1-X960njdFD2WcwFq','bytes':2701617525,
     'sha256':'bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d',
     'path':'third_party/intergen_base_20261008/checkpoints/intergen.ckpt'},
]


def write(path,value):
    pending=path.with_suffix('.tmp');pending.write_text(json.dumps(value,indent=2)+'\n');os.replace(pending,path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def fetch(asset,state):
    path=ROOT/asset['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.stat().st_size==asset['bytes'] and digest(path)==asset['sha256']
        return
    partial=path.with_suffix(path.suffix+'.partial')
    failures=0
    while (partial.stat().st_size if partial.exists() else 0)<asset['bytes']:
        start=partial.stat().st_size if partial.exists() else 0
        end=min(start+32*1024**2,asset['bytes'])-1
        try:
            with requests.get('https://drive.usercontent.google.com/download',
                    params={'id':asset['id'],'export':'download','confirm':'t'},
                    headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as response:
                response.raise_for_status()
                if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                    raise RuntimeError(f'Expected precise HTTP206 range, got {response.status_code} {response.headers.get("Content-Range")}')
                with partial.open('ab') as target:
                    for block in response.iter_content(1024**2):target.write(block)
            assert partial.stat().st_size==end+1
            failures=0
            state.update(asset=asset['name'],downloaded=partial.stat().st_size,total=asset['bytes'],updated=time.time())
            write(CONTROL/'STATUS.json',state)
        except Exception:
            failures+=1
            if failures>=6:raise
            time.sleep(min(5*failures,30))
    assert partial.stat().st_size==asset['bytes'] and digest(partial)==asset['sha256'],f'Pinned {asset["name"]} SHA mismatch; preserve partial for inspection'
    os.replace(partial,path)


def main():
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud-only baseline staging')
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'running','pid':os.getpid(),'started':time.time(),'verified':[],'gpu_work':False,'local_checkpoint_downloads':False}
        write(CONTROL/'STATUS.json',state)
        try:
            for asset in ASSETS:
                fetch(asset,state)
                state['verified'].append(asset)
                write(CONTROL/'STATUS.json',state)
            state.update(status='complete',ended=time.time());write(CONTROL/'STATUS.json',state)
        except Exception as error:
            state.update(status='failed',error=repr(error),ended=time.time());write(CONTROL/'STATUS.json',state)
            raise


if __name__=='__main__':main()
