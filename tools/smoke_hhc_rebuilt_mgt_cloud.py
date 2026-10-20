"""Real frozen-VQ/full-MGT cloud smoke; no FID or quality claims from 2 updates."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import clip
import h5py
import numpy as np
import torch
from torch.utils.data import default_collate
from dataset.dataset_interx_tokenize import deterministic_cache_crop
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset
from dataset.interhuman_hhc_adapter import OfficialInterHumanAdapter
from models.interhuman_evaluator_wrapper import InterHumanEvaluatorWrapper
from models.mgt_interhuman_smpl import SMPLMGT
from models.mgt_interhuman_smpl_cfg import configure_conditioning
from models.vqvae_interhuman_smpl_raw import NativeSMPLFK
import utils.interhuman_smpl_mgt as assets
from utils.hhc_frozen_vq import load_frozen_vq,file_sha
from utils.interhuman_mgt_training import interhuman_mgt_objective
from utils.motion_masks import sample_historical_mmm_corruption_with_kinematic_ratios
from utils.runtime import generate_src_mask
from tools.reconstruct_interhuman_raw_body import load_model


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--vq-spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if str(ROOT)!='/workspace/HHC':raise RuntimeError('GPU work only on cloud')
    args.output=args.output.resolve();assert args.output.is_relative_to(ROOT/'output');args.output.mkdir(exist_ok=False)
    status=args.output/'STATUS.json';state={'status':'initializing','pid':os.getpid(),'started':time.time(),
        'training_updates':2,'training_batch':2,'is_quality_evaluation':False,'checkpoint_saved':False}
    write(status,state)
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);os.nice(10)
        torch.cuda.set_per_process_memory_fraction(.5,0)
        torch.manual_seed(114514);np.random.seed(114514);random.seed(114514)
        data=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
        vq,mean,std,spec=load_frozen_vq(args.vq_spec,'cuda:0',data/'manifest.json')
        before={k:v.detach().cpu().clone() for k,v in vq.state_dict().items()}
        fk=NativeSMPLFK().cuda();rows=[]
        with h5py.File(data/'motions.h5') as archive:
            for name in (data/'split/train.txt').read_text().splitlines():
                if name not in archive or 'motion30' not in archive[name]:continue
                motion=archive[name]['motion30']
                if not 30<=len(motion)<1000:continue
                texts=[s.strip() for s in (data/'texts_processed'/f'{name}.txt').read_text().splitlines() if s.strip()]
                if not texts:continue
                n,start=deterministic_cache_crop(name,len(motion),4,114514)
                beta=torch.from_numpy(archive[name]['betas'][:]).cuda()
                with torch.no_grad():
                    native=fk.from_archive(torch.from_numpy(motion[start:start+n]).cuda(),beta).reshape(n,276)
                    tokens=vq(((native-mean)/std)[None],beta.reshape(1,20),torch.ones(1,2,dtype=torch.long,device='cuda'),type='encode')[0]
                rows.append({'name':name,'tokens':tokens.cpu(),'texts':texts,'beta':beta.reshape(20).cpu()})
                if len(rows)==2:break
        assert len(rows)==2
        dataset=assets.TokenData(rows);target,lengths,text,beta=default_collate([dataset[(i,114514+i)] for i in range(2)])
        clip_model,_=clip.load('ViT-B/32',device='cpu',jit=False);clip_model.eval().requires_grad_(False)
        global_text,word=assets.encode_text(clip_model,text,'cpu')
        net=configure_conditioning(SMPLMGT(vq).cuda(),'matched');net.train();vq.eval()
        assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in vq.state_dict().items())
        parameters=[p for p in net.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=.0002,betas=(.5,.9),weight_decay=1e-6)
        state.update(status='training',trainable_parameters=sum(p.numel() for p in parameters),vq_step=spec['step'],vq_sha256=spec['sha256']);write(status,state)
        training=[];target=target.cuda();beta=beta.cuda();lengths=lengths.cuda()
        for update in (1,2):
            valid=generate_src_mask(76,lengths);attention=generate_src_mask(76,lengths+1)
            corrupted,masked,_,_=sample_historical_mmm_corruption_with_kinematic_ratios(target,valid[:,:,None].expand_as(target),12288,pkeep=.5)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=net.forward_function(corrupted,global_text.cuda(),attention,word_emb=word.cuda(),betas=beta,
                    gender=torch.ones(2,2,dtype=torch.long,device='cuda'))
                result=interhuman_mgt_objective(logits,{'target':target,'valid_mask':valid[:,:,None].expand_as(target),'masked_mask':masked})
            assert torch.isfinite(result['loss']);optimizer.zero_grad(set_to_none=True);result['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            training.append({'update':update,'loss':float(result['loss']),'gradient_norm':float(norm)})
        net.eval();configure_conditioning(net,'matched',sampling=True)
        validation=InterHumanValidationRequestDataset();length_values=[r['processed_length'] for r in validation.requests]
        chosen=list(dict.fromkeys([int(np.argmin(length_values)),int(np.argmax(length_values)),1]))
        requests=[validation.requests[i] for i in chosen]
        assets.DATA=data
        scorer=assets.StreamingScorer(net,vq,mean,std,clip_model,None,None,OfficialInterHumanAdapter(),load_model(),'cuda:0','cpu')
        state.update(status='generating',validation_indices=chosen);write(status,state)
        generated=scorer.generate(requests,[r['sample_seed'] for r in requests],steps=50)
        assert generated.shape==(len(chosen),300,524) and torch.isfinite(generated).all()
        evaluator=InterHumanEvaluatorWrapper.from_official_checkpoint('cpu')
        lengths_cpu=torch.tensor([r['processed_length'] for r in requests])
        te,me=evaluator.get_co_embeddings(([r['identifier'] for r in requests],[r['caption'] for r in requests],
            generated[...,:262],generated[...,262:],lengths_cpu))
        assert torch.isfinite(te).all() and torch.isfinite(me).all()
        assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in vq.state_dict().items())
        state.update(status='passed',ended=time.time(),training=training,train_ids=[r['name'] for r in rows],
            validation_lengths=lengths_cpu.tolist(),generation_shape=list(generated.shape),
            evaluator_embedding_shape=list(me.shape),frozen_vq_unchanged=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            sources={p:file_sha(ROOT/p) for p in ('tools/smoke_hhc_rebuilt_mgt_cloud.py','utils/hhc_frozen_vq.py','models/mgt_interhuman_smpl.py')})
        write(status,state);print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='failed',error=repr(error),ended=time.time());write(status,state);raise


if __name__=='__main__':main()
