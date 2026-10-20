"""Native SMPL token cache and streaming official-coordinate MGT scoring."""
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import clip

from models.vqvae_interhuman_smpl_raw import NativeSMPLFK,VQSMPLRaw
from dataset.dataset_interx_tokenize import deterministic_cache_crop
from dataset.interhuman_smpl_pair import body192_from_motion
from dataset.interhuman_generated_h5 import mm_repeat_seed
from dataset.interhuman_fair_requests import InterHumanValidationRequestDataset,InterHumanFrozenRequestDataset
from tools.score_fair_interhuman_main import main_metrics_from_embeddings
from tools.score_fair_interhuman_mm import multimodality_from_embeddings,fixed_numpy_seed

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dataset/interhuman_smpl_interx_layout_v1_20261004'
VQ_RUN=Path('/home/wenning/HHC_interhuman_runs/vq_interhuman_smpl_raw_interx_handless_seed123_20261004')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()


def load_vq(device):
    checkpoint=torch.load(VQ_RUN/'best_fid.pth',map_location='cpu',weights_only=False)
    assert checkpoint['step']==185000
    model=VQSMPLRaw(SimpleNamespace(nb_code=12288,mu=.99))
    model.load_state_dict(checkpoint['net'],strict=True)
    model.requires_grad_(False).eval().to(device)
    return model,checkpoint['mean'].to(device),checkpoint['std'].to(device)


class TokenData(Dataset):
    def __init__(self,rows,seed=114514):
        self.rows=rows; self.base_seed=seed
        self.records=[(r['name'],False) for r in rows]

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        i,seed=index
        py=random.Random(seed); rng=np.random.default_rng(seed)
        row=self.rows[i]; tokens=row['tokens']
        text=py.choice(row['texts'])
        if rng.choice([False,False,True]) and len(tokens)>1:
            tokens=tokens[:-1] if rng.choice([True,False]) else tokens[1:]
        tokens=tokens[:75]
        target=torch.full((76,2),12289,dtype=torch.long)
        target[:len(tokens)]=tokens; target[len(tokens)]=12288
        return target,len(tokens),text,row['beta']


@torch.no_grad()
def build_tokens(vq,mean,std,device,path):
    signature={'vq':sha(VQ_RUN/'best_fid.pth'),'data':sha(DATA/'manifest.json'),
               'source':sha(__file__),'seed':114514,'crop':'Inter-X single/single/double cache; min30 max999; head token truncation75'}
    if path.exists():
        cache=torch.load(path,map_location='cpu',weights_only=False)
        assert cache['signature']==signature
        return TokenData(cache['rows'])
    fk=NativeSMPLFK().to(device); rows=[]
    with h5py.File(DATA/'motions.h5','r') as h:
        for name in (DATA/'split/train.txt').read_text().splitlines():
            if name not in h or 'motion30' not in h[name]: continue
            motion=h[name]['motion30']
            if not 30<=len(motion)<1000: continue
            texts=[line.strip() for line in (DATA/'texts_processed'/f'{name}.txt').read_text().splitlines()]
            if not texts: continue
            length,start=deterministic_cache_crop(name,len(motion),4,114514)
            beta=torch.from_numpy(h[name]['betas'][:]).to(device)
            x=fk.from_archive(torch.from_numpy(motion[start:start+length]).to(device),beta).reshape(length,276)
            token=vq(((x-mean)/std)[None],beta.reshape(1,20),torch.ones(1,2,dtype=torch.long,device=device),type='encode')[0]
            rows.append({'name':name,'tokens':token.cpu(),'texts':texts,'beta':beta.reshape(20).cpu()})
    pending=path.with_suffix('.tmp'); torch.save({'signature':signature,'rows':rows},pending); pending.replace(path)
    return TokenData(rows)


@torch.no_grad()
def encode_text(model,texts,device):
    tokens=clip.tokenize(list(texts),truncate=True).to(device)
    word=model.token_embedding(tokens).type(model.dtype)+model.positional_embedding.type(model.dtype)
    word=model.transformer(word.permute(1,0,2))
    word=model.ln_final(word).permute(1,0,2).float()
    return model.encode_text(tokens).float(),word


class StreamingScorer:
    def __init__(self,net,vq,mean,std,clip_model,evaluator,metrics,adapter,smpl,device,text_device):
        self.net,self.vq,self.mean,self.std=net,vq,mean,std
        self.clip,self.evaluator,self.metrics=clip_model,evaluator,metrics
        self.adapter,self.smpl=adapter,smpl
        self.device,self.text_device=device,text_device
        self.fk=NativeSMPLFK().to(device)
        with h5py.File(DATA/'motions.h5','r') as h:
            self.betas={name:torch.from_numpy(h[name]['betas'][:]) for name in h}

    @torch.no_grad()
    def generate(self,requests,seeds,steps=50):
        if self.net.conditioning_contract!='paper_text_cfg' or self.net.trans_base.conditioning_contract!='paper_text_cfg':
            raise ValueError('Generation CFG must retain requested shape: use paper_text_cfg')
        beta=torch.stack([self.betas[r['identifier']] for r in requests]).reshape(-1,20).to(self.device)
        gender=torch.ones(len(requests),2,dtype=torch.long,device=self.device)
        lengths=torch.tensor([r['processed_length'] for r in requests],device=self.device)
        # FK/online conversion removes one frame. Generate enough native frames
        # for every requested valid frame, never fill a valid tail with zeros.
        token_lengths=torch.div(lengths+4,4,rounding_mode='floor')
        text,word=encode_text(self.clip,[r['caption'] for r in requests],self.text_device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            codes=self.net.sample_intra_cas(text.to(self.device),word.to(self.device),beta,gender,
                                           token_lengths*4,seeds,max_steps=steps)
        output=torch.zeros(len(requests),300,524)
        for i,(request,n) in enumerate(zip(requests,token_lengths.tolist())):
            decoded=self.vq(codes[i:i+1,:n],beta[i:i+1],gender[i:i+1],type='decode')[0]
            raw=(decoded*self.std+self.mean).reshape(-1,23,12)
            raw=self.fk.to_archive(raw,beta[i])[:int(lengths[i])+1].cpu()
            body=body192_from_motion(raw,beta[i].reshape(2,10).cpu(),self.smpl).numpy()
            if request.get('left_right_swap',False):
                body=np.stack([self.adapter.data_utils.swap_left_right(p,22) for p in body])
            pair=self.adapter.process_loaded_pair(*body,anchor_person=request['anchor_person'],
                      output_person_swap=request['output_person_swap'],normalize=True)
            features=np.concatenate(pair,axis=-1)
            assert len(features)==int(lengths[i]) and np.isfinite(features).all()
            output[i,:len(features)]=torch.from_numpy(features)
        return output

    @torch.no_grad()
    def main_score(self,dataset,seed=20260130,steps=50,microbatch=8,embedding_path=None):
        texts_out=[]; pred_out=[]; gt_out=[]
        for begin in range(0,len(dataset),96):
            requests=dataset.requests[begin:begin+96]
            actual=[dataset[i] for i in range(begin,begin+len(requests))]
            pred=[]
            for offset in range(0,len(requests),microbatch):
                chunk=requests[offset:offset+microbatch]
                pred.append(self.generate(chunk,[mm_repeat_seed(r['sample_seed'],0) for r in chunk],steps))
            pred=torch.cat(pred)
            names=[r[0] for r in actual]; texts=[r[1] for r in actual]
            lengths=torch.tensor([r[4] for r in actual])
            gt1=torch.stack([r[2] for r in actual]); gt2=torch.stack([r[3] for r in actual])
            te,pe=self.evaluator.get_co_embeddings((names,texts,pred[...,:262],pred[...,262:],lengths))
            ge=self.evaluator.get_motion_embeddings((names,texts,gt1,gt2,lengths))
            texts_out.append(te.cpu().numpy()); pred_out.append(pe.cpu().numpy()); gt_out.append(ge.cpu().numpy())
        text,pred,gt=map(np.concatenate,(texts_out,pred_out,gt_out))
        if embedding_path is not None: np.savez_compressed(embedding_path,text=text,pred=pred,gt=gt)
        return main_metrics_from_embeddings(text,pred,gt,self.metrics,batch_size=96,random_seed=seed)

    @torch.no_grad()
    def mm_score(self,payload,replication=0,steps=50,microbatch=8,embedding_path=None):
        repeats=int(payload['mm_repeats']); all_embeddings=[]
        for request in payload['requests']:
            embeddings=[]
            for start in range(0,repeats,microbatch):
                indices=list(range(start,min(start+microbatch,repeats)))
                pred=self.generate([request]*len(indices),[mm_repeat_seed(request['sample_seed'],i) for i in indices],steps)
                batch=([request['identifier']]*len(indices),[request['caption']]*len(indices),
                       pred[...,:262],pred[...,262:],torch.tensor([request['processed_length']]*len(indices)))
                # The released wrapper reverse-sorts equal lengths. Undo each
                # microbatch sort, then apply the full-prompt order once.
                value=self.evaluator.get_motion_embeddings(batch).cpu().numpy()
                align=np.argsort(np.asarray(batch[-1]))[::-1].copy()
                embeddings.append(value[np.argsort(align)])
            whole=np.concatenate(embeddings)
            all_embeddings.append(whole[np.argsort(np.full(repeats,request['processed_length']))[::-1]])
        embeddings=np.stack(all_embeddings)
        if embedding_path is not None: np.savez_compressed(embedding_path,embeddings=embeddings)
        if repeats==30: return multimodality_from_embeddings(embeddings,self.metrics,random_seed=1234+replication)
        with fixed_numpy_seed(20260130): return float(self.metrics.calculate_multimodality(embeddings,10,6,2))
