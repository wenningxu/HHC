"""Acquire the author's pinned TIMotion source and public weight on cloud CPU.

The release has no published SHA. Freeze the first acquired bytes and report
that limitation; do not pretend the locally computed hash is author verified.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'output/timotion_restore_20261016'
REVISION='172580fd7f2551a7cc10fde1f15ff73315742e65'
ASSET_ID='15WE1kiITcrwva3QTGW86plgxA8zmufq4'
SIZE=1430359445


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(state):
    tmp=CONTROL/'STATUS.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,CONTROL/'STATUS.json')


def main():
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud only; no local checkpoint acquisition')
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'source','pid':os.getpid(),'started':time.time(),'gpu_work':False,
               'local_checkpoint_downloads':False,'source_revision':REVISION,
               'release_url':f'https://drive.google.com/file/d/{ASSET_ID}/view',
               'expected_bytes':SIZE,'author_published_checksum':None}
        write(state)
        try:
            source=ROOT/'third_party/TIMotion'
            if not source.exists():
                subprocess.run(['git','init',str(source)],check=True)
                subprocess.run(['git','remote','add','origin','https://github.com/AIGC-Explorer/TIMotion.git'],cwd=source,check=True)
                subprocess.run(['git','fetch','--depth','1','origin',REVISION],cwd=source,check=True)
                subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=source,check=True)
            assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==REVISION
            path=source/'checkpoints/timotion.ckpt';path.parent.mkdir(exist_ok=True)
            receipt=CONTROL/'weight_receipt.json'
            if path.exists():
                previous=json.loads(receipt.read_text())
                assert path.stat().st_size==SIZE and sha(path)==previous['acquired_sha256']
            else:
                partial=path.with_suffix('.ckpt.partial')
                failures=0
                while (partial.stat().st_size if partial.exists() else 0)<SIZE:
                    start=partial.stat().st_size if partial.exists() else 0;end=min(start+32*1024**2,SIZE)-1
                    try:
                        with requests.get('https://drive.usercontent.google.com/download',
                                params={'id':ASSET_ID,'export':'download','confirm':'t'},
                                headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as response:
                            response.raise_for_status()
                            assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {start}-')
                            with partial.open('ab') as stream:
                                for block in response.iter_content(1024**2):stream.write(block)
                        assert partial.stat().st_size==end+1
                        failures=0;state.update(status='weight',downloaded=end+1,updated=time.time());write(state)
                    except Exception:
                        failures+=1
                        if failures>=6:raise
                        time.sleep(min(5*failures,30))
                assert partial.stat().st_size==SIZE
                digest=sha(partial)
                # Persist the acquisition receipt before the final rename.
                receipt.write_text(json.dumps({**state,'acquired_sha256':digest,
                    'verification':'public author link + live metadata size; SHA computed on acquisition, no author checksum',
                    'source_path':str(source),'checkpoint_path':str(path)},indent=2)+'\n')
                os.replace(partial,path)
            state.update(status='complete',ended=time.time(),checkpoint_path=str(path),acquired_sha256=sha(path));write(state)
        except Exception as error:
            state.update(status='failed',error=repr(error),ended=time.time());write(state);raise


if __name__=='__main__':main()
