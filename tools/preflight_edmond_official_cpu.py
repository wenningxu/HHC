"""Check the copied original dataset/runtime and source catalogue on Edmond CPU."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT=Path('/workspace/HHC')
SOURCE=ROOT/'third_party/intergen_official_baseline_20261017'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    launch=json.loads((ROOT/'output/intergen_official_baseline_20261017/LAUNCH.json').read_text())
    provenance=json.loads((ROOT/'output/intergen_official_baseline_setup_20261017/SOURCE_PROVENANCE.json').read_text())
    for name,expected in provenance['tracked_files'].items():
        expected=launch.get('source_changes',{}).get(name,{}).get('after',expected)
        assert sha(SOURCE/name)==expected,name
    sys.path.insert(0,str(SOURCE));os.chdir(SOURCE)
    import torch
    import lightning.pytorch as pl
    from configs import get_config
    from datasets import DataModule
    assert pl.__version__=='1.9.1'
    data=get_config('configs/datasets.yaml').interhuman
    config=get_config('configs/train.yaml')
    assert config.TRAIN.BATCH_SIZE==32 and config.TRAIN.EPOCH==2000
    module=DataModule(data,config.TRAIN.BATCH_SIZE,config.TRAIN.NUM_WORKERS)
    module.setup('fit')
    count=len(module.train_dataset)
    updates=len(module.train_dataloader())
    assert count==11274 and updates==352 and not torch.cuda.is_initialized()
    out=ROOT/'output/intergen_official_baseline_20261017/edmond_handoff_20261020'
    out.mkdir(exist_ok=True)
    result={'status':'original_dataset_cpu_verified','dataset_rows':count,'updates_per_epoch':updates,
            'source_files_verified':len(provenance['tracked_files']),'torch':torch.__version__,
            'lightning':pl.__version__,'cuda_initialized':False,'time':time.time()}
    with (out/'destination_preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':
    main()
