"""Pinned public TIMotion Transformer+LPA on frozen validation, cloud only."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import h5py
import numpy as np
import torch
import yaml
from yacs.config import CfgNode

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset,sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--limit',type=int,default=480)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('Cloud execution only')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    source=ROOT/'third_party/TIMotion'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert revision=='172580fd7f2551a7cc10fde1f15ff73315742e65'
    assert not subprocess.check_output(['git','diff','--name-only','HEAD','--','models','configs','utils'],cwd=source,text=True).strip()
    receipt=json.loads((ROOT/'output/timotion_restore_20261016/weight_receipt.json').read_text())
    checkpoint=source/'checkpoints/timotion.ckpt'
    digest=sha256_file(checkpoint)
    assert checkpoint.stat().st_size==1430359445 and digest==receipt['acquired_sha256']
    data=InterHumanValidationRequestDataset()
    assert 1<=args.limit<=len(data)
    (source/'data').mkdir(exist_ok=True)
    for name,value in [('global_mean.npy',data.adapter.mean),('global_std.npy',data.adapter.std)]:
        target=source/'data'/name
        if not target.exists():target.symlink_to(ROOT/'third_party/intergen_base_20261008/data'/name)
        assert np.array_equal(np.load(target),value)
    os.chdir(source);sys.path.insert(0,str(source))
    from models import TIMotion
    assert Path(sys.modules[TIMotion.__module__].__file__).resolve().is_relative_to(source)
    cfg=CfgNode(yaml.safe_load((source/'configs/model.yaml').read_text()))
    # Exact defaults in the pinned tools/eval.py, plus the README's --LPA.
    cfg.NUM_HEADS=16;cfg.NUM_LAYERS=5;cfg.LATENT_DIM=512
    cfg.LPA=True;cfg.conv_layers=1;cfg.dilation_rate=1;cfg.norm='AdaLN'
    assert cfg.STRATEGY=='ddim50' and cfg.CFG_WEIGHT==3.5
    model=TIMotion(cfg).eval()
    release=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state={key.removeprefix('model.'):value for key,value in release['state_dict'].items()}
    loaded=model.load_state_dict(state,strict=True)
    metadata={'model':'TIMotion public Transformer+LPA','source_commit':revision,
        'checkpoint_sha256':digest,'checkpoint_bytes':checkpoint.stat().st_size,
        'author_published_checksum':None,'missing_keys':loaded.missing_keys,'unexpected_keys':loaded.unexpected_keys,
        'state_tensors':len(state),'config':yaml.safe_load(cfg.dump()),
        'request_sha256':data.payload['request_sha256'],'split':'validation','limit':args.limit,'frame_cap':0,
        'seed':20261006,'sampler':{'strategy':'ddim50','cfg_weight':3.5,'batch_size':1},
        'protocol':'frozen validation requests; official unrounded length; raw denormalized 262D; no interventions',
        'is_formal_test':False,'script_sha256':sha256_file(Path(__file__)),
        'device':'cpu' if args.inspect_only else args.device,'pid':os.getpid()}
    args.output=args.output if args.output.is_absolute() else ROOT/args.output
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.inspect_only:
        metadata['status']='strict_cpu_load_passed'
        args.output.write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True);return
    del release,state
    device=torch.device(args.device)
    assert device.type=='cuda' and torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.5,0)
    model=model.to(device)
    random.seed(20261006);np.random.seed(20261006);torch.manual_seed(20261006)
    started=time.monotonic()
    with h5py.File(args.output,'x') as cache,torch.inference_mode():
        cache.attrs['metadata']=json.dumps(metadata);cache.attrs['status']='running'
        for i in range(args.limit):
            name,text,a,b,n=data[i]
            prediction=model.forward_test({'text':[text],'motion_lens':torch.tensor([n],device=device)})['output'][0].reshape(n,2,262)
            assert torch.isfinite(prediction).all()
            group=cache.create_group(str(i));group.attrs['request']=json.dumps(data.requests[i])
            group.attrs['caption']=text;group.attrs['gt_length']=n;group.attrs['generated_length']=n
            for key,value in [('gt',torch.stack([a,b],dim=1)[:n]),('gen',prediction)]:
                raw=value.cpu().numpy()*data.adapter.std+data.adapter.mean
                group.create_dataset(key,data=raw,compression='gzip',compression_opts=1)
            cache.flush()
            print(json.dumps({'generated':i+1,'length':n,'elapsed_s':time.monotonic()-started}),flush=True)
        cache.attrs['status']='complete';cache.attrs['elapsed_s']=time.monotonic()-started


if __name__=='__main__':main()
