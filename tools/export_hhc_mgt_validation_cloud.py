"""Independent frozen-val MGT export, including a text-only zero-shape policy.

Requires completed training. Reads the final checkpoint by default; no test
data, GT motion, or GT shape enters the zero-shape generation path.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import clip
import h5py
import numpy as np
import torch
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_generated_h5 import MEAN_SHA256,STD_SHA256,InterHumanGeneratedH5Dataset,mm_repeat_seed
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from models.mgt_interhuman_smpl import SMPLMGT
from models.mgt_interhuman_smpl_cfg import configure_conditioning
from tools.reconstruct_interhuman_raw_body import load_model
from utils.hhc_frozen_vq import load_frozen_vq,file_sha
from utils.interhuman_smpl_mgt import StreamingScorer
import utils.interhuman_smpl_mgt as assets


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mgt-run',type=Path,required=True)
    parser.add_argument('--vq-spec',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shape-policy',choices=['zero','requested'],default='zero')
    parser.add_argument('--selection',choices=['final','val_best'],default='final');args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('GPU work only on cloud')
    manifest=json.loads((args.mgt_run/'training_manifest.json').read_text());contract=manifest['contract']
    assert manifest['status']=='pilot_complete_validation_only' and manifest['step']==contract['steps']==40000
    assert contract['vq_spec_sha256']==file_sha(args.vq_spec)
    for p,digest in contract['sources'].items():assert file_sha(ROOT/p)==digest,'Training source changed: '+p
    checkpoint=args.mgt_run/('latest.pth' if args.selection=='final' else 'best_fid.pth')
    checkpoint_sha=file_sha(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['contract_sha256']==manifest['contract_sha256']
    selected_step=saved['step'];assert selected_step==(40000 if args.selection=='final' else manifest['best_step'])
    model_state=saved['model'];del saved
    args.output=args.output.resolve();assert args.output.is_relative_to(ROOT/'output');args.output.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);os.nice(10)
    torch.cuda.set_per_process_memory_fraction(.35,0)
    native=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004';assets.DATA=native
    vq,mean,std,spec=load_frozen_vq(args.vq_spec,'cuda:0',native/'manifest.json')
    net=SMPLMGT(vq).cuda();loaded=net.load_state_dict(model_state,strict=False);del model_state
    assert not loaded.unexpected_keys and all(k.startswith('trans_base.vqvae.') for k in loaded.missing_keys)
    net.eval().requires_grad_(False);configure_conditioning(net,contract['conditioning_arm'],sampling=True)
    clip_model,_=clip.load('ViT-B/32',device='cpu',jit=False);clip_model.eval().requires_grad_(False)
    data=InterHumanValidationRequestDataset()
    scorer=StreamingScorer(net,vq,mean,std,clip_model,None,None,OfficialInterHumanAdapter(),load_model(),'cuda:0','cpu')
    if args.shape_policy=='zero':scorer.betas={r['identifier']:torch.zeros(2,10) for r in data.requests}
    metadata={'method':'hhc_native_smpl_mgt','mgt_checkpoint_sha256':checkpoint_sha,'mgt_step':selected_step,
        'vq_checkpoint_sha256':spec['sha256'],'vq_step':spec['step'],'vq_spec_sha256':file_sha(args.vq_spec),
        'shape_policy':args.shape_policy,'gt_shape_used':args.shape_policy=='requested',
        'shape_policy_scope':'Both MGT conditions, VQ decoder and native FK use the same selected betas',
        'selection':args.selection,'selection_protocol':'fixed final40000' if args.selection=='final' else 'val-best selected under GT-shape training monitor',
        'conditioning_arm':contract['conditioning_arm'],'token_window':contract['token_window_policy'],
        'sampling_steps':50,'cfg':3.5,'temperature':.5,'sampling_seed':'mm_repeat_seed(request.sample_seed,0)',
        'exporter_sha256':file_sha(Path(__file__)),'is_formal_test':False}
    temporary=args.output/'generations.partial.h5';destination=args.output/'generations.h5';started=time.time()
    with h5py.File(temporary,'x') as out:
        out.attrs.update(schema='interhuman_fair_generation_v1',status='running',split='val',
            request_file_sha256=data.manifest['request']['sha256'],request_content_sha256=data.payload['request_sha256'],
            normalization_mean_sha256=MEAN_SHA256,normalization_std_sha256=STD_SHA256,metadata=json.dumps(metadata))
        strings=h5py.string_dtype('utf-8')
        out.create_dataset('names',data=[r['identifier'] for r in data.requests],dtype=strings)
        out.create_dataset('texts',data=[r['caption'] for r in data.requests],dtype=strings)
        out.create_dataset('lengths',data=[r['processed_length'] for r in data.requests],dtype=np.int64)
        for key in ('motion1','motion2'):out.create_dataset(key,(480,300,262),dtype=np.float32,chunks=(1,300,262),compression='lzf')
        for begin in range(0,480,4):
            requests=data.requests[begin:begin+4]
            prediction=scorer.generate(requests,[mm_repeat_seed(r['sample_seed'],0) for r in requests],steps=50)
            assert torch.isfinite(prediction).all()
            out['motion1'][begin:begin+len(requests)]=prediction[...,:262].numpy()
            out['motion2'][begin:begin+len(requests)]=prediction[...,262:].numpy()
            if (begin+4)%96==0:print(json.dumps({'generated':begin+4,'shape_policy':args.shape_policy,'elapsed_s':time.time()-started}),flush=True)
        out.attrs['status']='complete'
    assert file_sha(checkpoint)==checkpoint_sha
    os.replace(temporary,destination)
    verified=InterHumanGeneratedH5Dataset(destination,data.request_path,request_entry=data.manifest['request'],formal=False);verified.close()
    result={'status':'complete','samples':480,'h5':str(destination),'h5_sha256':file_sha(destination),
            'metadata':metadata,'gpu_peak_gib':torch.cuda.max_memory_allocated()/1024**3,'elapsed_s':time.time()-started}
    (args.output/'STATUS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
