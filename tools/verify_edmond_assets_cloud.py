"""Hash static model/data assets on the source; verify the same manifest on Edmond."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/workspace/HHC')
MANIFEST=ROOT/'output/edmond_migration_20261020/static_assets_manifest.json'

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:
        files=json.loads(MANIFEST.read_text())
    else:
        files={}
        suffixes={'.pt','.pth','.ckpt','.pkl','.npy','.npz','.h5'}
        for base in [ROOT,Path('/dev/shm/hhc_inputs'),Path('/root/.cache/clip')]:
            for p in base.rglob('*'):
                if p.is_file() and not p.is_symlink() and p.suffix in suffixes:
                    if '/official_intergen_scratch_20261017/model/' in str(p):continue
                    if '/edmond_handoff_20261020/' in str(p):continue
                    files[str(p)]={}
    cache={}
    total=0
    for name,expected in files.items():
        p=Path(name);s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity not in cache:cache[identity]=digest(p)
        actual={'bytes':s.st_size,'sha256':cache[identity]}
        if args.verify:assert actual==expected,name
        else:files[name]=actual
        total+=s.st_size
    if args.verify:
        receipt={'status':'static_assets_sha256_verified','files':len(files),'logical_bytes':total,
                 'manifest_sha256':digest(MANIFEST),'time':time.time()}
        (MANIFEST.parent/'static_assets_destination_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))
    else:
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        with MANIFEST.open('x') as f:json.dump(files,f,indent=2);f.write('\n')
        print(json.dumps({'files':len(files),'logical_bytes':total,'manifest_sha256':digest(MANIFEST)}))

if __name__=='__main__':
    main()
