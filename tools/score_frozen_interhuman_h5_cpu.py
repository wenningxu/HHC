"""Shared raw-coordinate, per-request CPU InterCLIP scoring for bound H5 files.

Matches the three published-baseline validation runs' embedding batching,
request order, retrieval batches and diversity seed. No model adaptation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset,InterHumanFrozenRequestDataset
from dataset.interhuman_generated_h5 import InterHumanGeneratedH5Dataset,sha256_file
from models.interhuman_evaluator_wrapper import load_official_interclip,load_official_metrics_module,evaluator_provenance
from tools.score_fair_interhuman_main import main_metrics_from_embeddings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--manifest',type=Path);parser.add_argument('--replication',type=int)
    parser.add_argument('--reference-embeddings',type=Path);args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.embeddings.npz').exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if os.name!='nt':os.nice(10)
    if args.split=='val':
        if args.replication is not None:parser.error('val has no replication selector')
        manifest=args.manifest or ROOT/'output/interhuman_validation_requests/manifest.json'
        data=InterHumanValidationRequestDataset(manifest)
        entry=data.manifest['request'];request_path=data.request_path;expected=480;seed=20261006
    else:
        if args.replication is None:parser.error('test requires a frozen replication')
        manifest=args.manifest or ROOT/'output/interhuman_fair_requests_v3/manifest.json'
        data=InterHumanFrozenRequestDataset(args.replication,manifest)
        entries=json.loads(manifest.read_text())['replications'];entry=next(r for r in entries if r['replication']==args.replication)
        request_path=Path(entry['path']);expected=1056;seed=1234
    generated=InterHumanGeneratedH5Dataset(args.generations,request_path,request_entry=entry,formal=args.split=='test')
    assert len(data)==len(generated)==expected
    assert generated.metadata['status']=='complete' and generated.metadata['split']==args.split
    model,_=load_official_interclip('cpu');model.requires_grad_(False).eval()
    outputs={key:[] for key in ('GT','raw','text')};started=time.time()
    with torch.inference_mode():
        for i in range(expected):
            name,text,a,b,n=data[i];pn,pt,p,q,ln=generated[i]
            assert name==pn and text==pt and n==ln
            for first,second,key in ((a,b,'GT'),(p,q,'raw')):
                normalized=torch.stack((first,second),dim=1)[:n].numpy()
                raw=normalized*data.adapter.std+data.adapter.mean
                value=torch.from_numpy(raw.astype(np.float32,copy=False)).flatten(1)[None]
                z=model.encode_motion({'motions':value,'motion_lens':torch.tensor([n])})['motion_emb']
                assert z.shape==(1,512) and torch.isfinite(z).all();outputs[key].append(z.numpy()[0])
            outputs['text'].append(model.encode_text({'text':[text]})['text_emb'].numpy()[0])
            if (i+1)%96==0:print(json.dumps({'encoded':i+1,'elapsed_s':time.time()-started}),flush=True)
    arrays={key:np.stack(value) for key,value in outputs.items()};comparison=None
    if args.reference_embeddings:
        with np.load(args.reference_embeddings) as reference:
            comparison={key:{'equal':bool(np.array_equal(arrays[key],reference[key])),
                'max_abs':float(np.abs(arrays[key]-reference[key]).max())} for key in ('GT','text')}
        assert all(r['equal'] for r in comparison.values()),'Reference GT/text embeddings changed'
    metrics=main_metrics_from_embeddings(arrays['text'],arrays['raw'],arrays['GT'],load_official_metrics_module(),batch_size=96,random_seed=seed)
    args.output.parent.mkdir(parents=True,exist_ok=True);embeddings=args.output.with_suffix('.embeddings.npz')
    with embeddings.open('xb') as stream:np.savez_compressed(stream,**arrays)
    report={'status':'complete','split':args.split,'replication':args.replication,'samples':expected,
        'protocol':'raw 262D; CPU InterCLIP per-request encoding; canonical request order; within-batch96 retrieval',
        'diversity_seed':seed,'metrics':metrics,'generation_metadata':generated.metadata,
        'generation_sha256':sha256_file(args.generations),'request_sha256':entry['request_sha256'],
        'evaluator':evaluator_provenance(include_checkpoint_sha256=True),'embeddings_sha256':sha256_file(embeddings),
        'reference_embeddings_sha256':sha256_file(args.reference_embeddings) if args.reference_embeddings else None,
        'reference_gt_text':comparison,'script_sha256':sha256_file(Path(__file__)),'elapsed_s':time.time()-started}
    generated.close()
    pending=args.output.with_suffix('.tmp');pending.write_text(json.dumps(report,indent=2)+'\n');os.replace(pending,args.output)
    print(json.dumps({'status':'complete','metrics':metrics,'reference_gt_text':comparison}),flush=True)


if __name__=='__main__':main()
