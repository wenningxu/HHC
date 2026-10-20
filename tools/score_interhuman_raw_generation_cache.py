"""Shared frozen-480val scorer for released InterGen / InterMask raw caches.

Keeps method-native generation lengths, including InterMask floor(L/4)*4.
No feature intervention or evaluator retraining. Intended to run on cloud CPU.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset,sha256_file
from models.interhuman_evaluator_wrapper import load_official_interclip,load_official_metrics_module,evaluator_provenance
from tools.score_fair_interhuman_main import main_metrics_from_embeddings
SCRIPT_SHA256=sha256_file(Path(__file__))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=['intermask','intergen','timotion'],required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2)
    data=InterHumanValidationRequestDataset()
    assert len(data)==480
    model,_=load_official_interclip('cpu')
    model.requires_grad_(False)
    text=[];gt=[];gen=[];lengths=[]
    started=time.time()
    with h5py.File(args.cache) as cache,torch.inference_mode():
        assert cache.attrs['status']=='complete' and len(cache)==480
        metadata=json.loads(cache.attrs['metadata'])
        assert metadata['request_sha256']==data.payload['request_sha256'] and metadata['limit']==480
        assert metadata['split']=='validation' and metadata['frame_cap']==0
        if args.method=='intermask':
            assert metadata['vq_sha256']=='af57bbdde94ed4d4a01f3ba5f9fe24f0012c2a37b3e38c2fa57a2a91ebcc1272'
            assert metadata['transformer_sha256']=='392ca8c9e42c2c23aeb97ba7b810e8fa4bdcc5f4839a5a909be2f56bf1698665'
        elif args.method=='intergen':assert metadata['checkpoint_sha256']=='bab341123c27af9d8c3ee7ebc43c2d341bfd0f4103425d06f8af6c717a086f8d'
        else:
            assert metadata['checkpoint_sha256']=='f364605f7e67cc272dc31810f384ff0dedb564331442257bf776f8d7285ffcee'
            assert metadata['model']=='TIMotion public Transformer+LPA'
        for i,request in enumerate(data.requests):
            group=cache[str(i)]
            assert json.loads(group.attrs['request'])==request
            g,p=group['gt'][:],group['gen'][:]
            n=request['processed_length']
            expected_pred=n//4*4 if args.method=='intermask' else n
            assert g.shape==(n,2,262) and p.shape==(expected_pred,2,262)
            assert np.isfinite(g).all() and np.isfinite(p).all()
            _,caption,a,b,length=data[i]
            reference=np.stack([a[:length].numpy(),b[:length].numpy()],axis=1)*data.adapter.std+data.adapter.mean
            assert np.allclose(g,reference,rtol=0,atol=2e-6)
            for values,outputs in [(g,gt),(p,gen)]:
                value=torch.from_numpy(values.astype(np.float32,copy=False)).flatten(1)[None]
                z=model.encode_motion({'motions':value,'motion_lens':torch.tensor([len(values)])})['motion_emb']
                assert z.shape==(1,512) and torch.isfinite(z).all()
                outputs.append(z.numpy()[0])
            text.append(model.encode_text({'text':[caption]})['text_emb'].numpy()[0])
            lengths.append({'gt':n,'generated':expected_pred})
            if (i+1)%96==0:print(json.dumps({'encoded':i+1,'elapsed_s':time.time()-started}),flush=True)
    arrays={'GT':np.stack(gt),'raw':np.stack(gen),'text':np.stack(text)}
    metrics=main_metrics_from_embeddings(arrays['text'],arrays['raw'],arrays['GT'],load_official_metrics_module(),batch_size=96,random_seed=20261006)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    embeddings=args.output.with_suffix('.embeddings.npz')
    with embeddings.open('xb') as f:np.savez_compressed(f,**arrays)
    report={'status':'complete','method':args.method,'split':'validation','samples':480,
            'protocol':'shared frozen 480 requests; native method lengths; raw 262D; no interventions; official CPU InterCLIP; within-batch96 retrieval',
            'is_formal_test':False,'multimodality':'pending; requires separate repeated generations',
            'metrics':metrics,'metadata':metadata,'lengths':lengths,'request_sha256':data.payload['request_sha256'],
            'cache_sha256':sha256_file(args.cache),'evaluator':evaluator_provenance(include_checkpoint_sha256=True),
            'embeddings_sha256':sha256_file(embeddings),'script_sha256':SCRIPT_SHA256,'elapsed_s':time.time()-started}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'complete','metrics':metrics}),flush=True)


if __name__=='__main__':main()
