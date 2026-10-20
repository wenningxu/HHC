"""Fetch pinned fallback assets after the initial recovery backup completes."""
import json
import msvcrt
import os
from pathlib import Path
import time
import sys
from local_compute_control import download, event, atomic_json

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'migration/control'

def public_interclip(asset):
    """Resume the same partial using the official released evaluator only."""
    import hashlib
    sys.path.insert(0,str(ROOT/'migration/download_packages'))
    import requests
    os.environ.setdefault('HTTPS_PROXY','http://127.0.0.1:7892')
    target=ROOT/asset['local']
    partial=target.with_suffix(target.suffix+'.partial')
    size=asset['bytes']
    assert not partial.exists() or partial.stat().st_size<=size
    while not partial.exists() or partial.stat().st_size<size:
        start=partial.stat().st_size if partial.exists() else 0
        end=min(start+8*1024*1024,size)-1
        with requests.get('https://drive.usercontent.google.com/download',
            params={'id':'1bJv5lTP7otJleaBYZ2byjru_k_wCsGvH','export':'download','confirm':'t'},
            headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,90)) as response:
            response.raise_for_status()
            assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{size}'
            remaining=end-start+1
            with partial.open('ab') as out:
                for block in response.iter_content(1024*1024):
                    assert len(block)<=remaining
                    out.write(block);remaining-=len(block)
            assert remaining==0
        atomic_json(CONTROL/'interclip-public-status.json',{'status':'downloading','bytes':partial.stat().st_size,'total':size})
    with partial.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==asset['sha256'], 'Frozen evaluator SHA256 mismatch'
    os.replace(partial,target)
    receipt={'file':str(target),'remote':asset['remote'],'source_file_id':'1bJv5lTP7otJleaBYZ2byjru_k_wCsGvH',
             'sha256':asset['sha256'],'bytes':size,'verified_at':time.time(),'cloud_status':'pending'}
    atomic_json(target.with_suffix(target.suffix+'.receipt.json'),receipt)
    atomic_json(CONTROL/'interclip-public-status.json',dict(receipt,status='complete'))
    event('artifact_verified',**receipt)
    return target

def main():
    with (CONTROL/'runtime-assets.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1): lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        atomic_json(CONTROL/'runtime-assets.pid.json',{'pid':os.getpid(),'started':time.time()})
        initial=ROOT/'checkpoints/local_backups/intergen_scratch_20261010_step000000/recovery.pt.receipt.json'
        while not initial.exists(): time.sleep(3600)
        manifest=ROOT/'migration/runtime-assets.json'
        data=json.loads(manifest.read_text())
        for asset in data['assets']:
            while True:
                try:
                    path=download(asset['remote'],ROOT/asset['local'],asset['sha256'],asset['bytes'])
                    asset.update(status='verified_local',verified_at=time.time(),file=str(path))
                    atomic_json(manifest,data)
                    break
                except Exception as exc:
                    event('runtime_asset_download_retry',asset=asset['name'],error=str(exc))
                    if asset['name']=='interclip':
                        try:
                            path=public_interclip(asset)
                            asset.update(status='verified_local',verified_at=time.time(),file=str(path))
                            atomic_json(manifest,data)
                            break
                        except Exception as public_exc:
                            event('runtime_asset_public_retry',asset=asset['name'],error=str(public_exc))
                    time.sleep(3600)
        event('runtime_assets_verified_local',assets=[a['name'] for a in data['assets']])

if __name__=='__main__': main()
