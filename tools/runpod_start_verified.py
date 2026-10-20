"""Gate a Runpod scratch continuation on verified complete inputs, then exec training."""
import hashlib,json,os,pathlib,subprocess,sys,tarfile,time
ROOT=pathlib.Path('/workspace/HHC')
assets=[('/workspace/best.pth',729644528,'e432c7a2ed82ae3fbfbb7bcdb00525e241eee0e44d35297a9b23e66d0454c2dd'),
('/workspace/SMPL_FEMALE.pkl',247530000,'6d4a1791b6b94880397e1a3a4539b703a228d2150c57de7b288389a8115f4ef0'),
('/workspace/runpod-raw-20261015.tar.gz',642329772,'1246ce6cb461daae02063a84fb6df31b2e24511429af1ce2ad934c8dd55e419c')]
deadline=time.monotonic()+7200
while any(not pathlib.Path(p).exists() or pathlib.Path(p).stat().st_size!=size for p,size,_ in assets):
    if time.monotonic()>deadline:raise TimeoutError('Staging deadline exceeded')
    time.sleep(20)
for p,size,sha in assets:
    with open(p,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha,p
    print(json.dumps({'verified':p,'bytes':size,'sha256':sha}),flush=True)
fast=pathlib.Path('/dev/shm/hhc_inputs');fast.mkdir(parents=True,exist_ok=True)
with tarfile.open(assets[2][0]) as t:
    for member in t.getmembers():assert (fast/member.name).resolve().is_relative_to(fast.resolve())
    t.extractall(fast,filter='data')
for name in ('motions','annots','split'):
    canonical=ROOT/'dataset/interhuman_official'/name
    if not canonical.is_symlink():
        if canonical.exists():canonical.rename(canonical.with_name(name+f'.staged_original_{time.time_ns()}'))
        canonical.symlink_to(fast/'dataset/interhuman_official'/name,target_is_directory=True)
(ROOT/'body_models/smpl').mkdir(parents=True,exist_ok=True)
smpl=ROOT/'body_models/smpl/SMPL_FEMALE.pkl'
if not smpl.exists():os.link('/workspace/SMPL_FEMALE.pkl',smpl)
run=ROOT/'output/intergen_rotation_scratch_runpod_20261015';run.mkdir(parents=True,exist_ok=True)
for name in ('normalization.npz','normalization.json'):
    source=ROOT/'output/intergen_rotation_scratch_20261010'/name
    if not (run/name).exists():os.link(source,run/name)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'third_party/intergen_base_20261008',text=True).strip()
assert commit=='b1750df4a859443b8a69329f759ce3a7503ae179'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from dataset.interhuman_root_rotation import RootRotationData
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
import torch
torch.set_num_threads(2)
data=RootRotationData(); val=InterHumanValidationRequestDataset(adapter=data.adapter)
assert len(val)==480
for i in range(len(val)):val[i]
print(json.dumps({'frozen_validation_inputs_checked':len(val),'request_sha256':val.payload['request_sha256']}),flush=True)
while not (ROOT/'ASSETS_READY.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Frozen model staging deadline exceeded')
    time.sleep(20)
args=[sys.executable,'-u','tools/train_intergen_rotation.py','--scratch','--output',str(run),'--write-recovery-receipt']
if (run/'recovery.pt').exists():args+=['--resume']
else:args+=['--resume-weights-only','/workspace/best.pth']
os.execv(sys.executable,args)
