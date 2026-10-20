"""Historical Inter-X shape-conditioned MGT, with the hand stream removed."""
import torch
from torch import nn
from models.two_person_transformer import CrossCondTransBase, CrossCondTransHead, Text2Motion_Transformer
from models.motion_transformer import AdaBlock, Block_crossatt
from models.trans_encdec import PositionEmbeddingSine1D
from models.interhuman_transformer import InterHumanText2MotionTransformer
from utils.runtime import generate_src_mask, cosine_schedule


class SMPLTransBase(CrossCondTransBase):
    def __init__(self,vq,embed_dim=1024,layers=9,heads=16,dropout=.1,block_size=76):
        nn.Module.__init__(self)
        self.vqvae=vq
        self.num_vq=vq.num_code
        self.block_size=block_size
        self.conditioning_contract='historical_mmm_shape_dropout'
        self.shape_mask_prob=.2
        self.text_drop_prob=.1
        self.special=nn.ModuleList([nn.Embedding(3,d) for d in (96,64)])
        self.projection=nn.ModuleList([nn.Linear(d,embed_dim) for d in (96,64)])
        self.part_emb=nn.Embedding(2,embed_dim)
        self.pos_embed=PositionEmbeddingSine1D(embed_dim,0.,block_size,True)
        self.word_emb=nn.Linear(512,embed_dim)
        self.cond_emb=nn.Linear(512,embed_dim)
        self.gender_embedding=nn.Embedding(2,4)
        self.shape_projection=nn.Linear(20,40)
        self.condition_mlp=nn.Sequential(nn.Linear(48,embed_dim),nn.ReLU(),nn.Linear(embed_dim,embed_dim))
        self.cross_att=nn.ModuleList([Block_crossatt(embed_dim,block_size,heads,dropout,4) for _ in range(2)])
        self.blocks=nn.ModuleList([AdaBlock(embed_dim,block_size,heads,dropout,4) for _ in range(layers-2)])
        for name,module in self.named_children():
            if name!='vqvae': module.apply(self._init_weights)

    def forward(self,idxs,clip_feature,src_mask,word_emb,betas,gender,force_mask=False):
        assert idxs.ndim==3 and idxs.shape[-1]==2 and idxs.shape[1]<=self.block_size
        assert idxs.min()>=0 and idxs.max()<self.num_vq+3
        parts=[]
        for part,channel in enumerate((1,0)):
            ids=idxs[...,channel]; ordinary=ids<self.num_vq
            latent=torch.zeros(*ids.shape,(96,64)[channel],device=ids.device)
            quantizer=(self.vqvae.quantizer_upper,self.vqvae.quantizer_root)[channel]
            if ordinary.any(): latent[ordinary]=quantizer.dequantize(ids[ordinary]).detach()
            if (~ordinary).any(): latent[~ordinary]=self.special[channel](ids[~ordinary]-self.num_vq)
            value=self.projection[channel](latent)
            parts.append(value+self.pos_embed(value)+self.part_emb.weight[part])
        x=torch.stack(parts,dim=2).flatten(1,2)
        if self.conditioning_contract=='paper_text_cfg':
            word_emb,clip_feature=self.mask_text(word_emb,clip_feature,force_mask)
        word_emb=self.word_emb(word_emb)
        for block in self.cross_att: x=block(x,word_emb)
        if self.conditioning_contract=='historical_mmm_shape_dropout':
            clip_feature=self.mask_shape(clip_feature,force_mask)
        x=torch.cat([self.cond_emb(clip_feature)[:,None],x],dim=1)
        shape=self.shape_projection(self.mask_shape(betas,force_mask and self.conditioning_contract=='historical_mmm_shape_dropout'))
        condition=self.condition_mlp(torch.cat([self.gender_embedding(gender).reshape(-1,8),shape],dim=-1))
        for block in self.blocks: x=block(x,condition,src_mask)
        return x


class SMPLMGT(Text2Motion_Transformer):
    def __init__(self,vq,embed_dim=1024,layers=9,heads=16,dropout=.1,block_size=76):
        nn.Module.__init__(self)
        self.n_head=heads; self.num_vq=vq.num_code; self.block_size=block_size
        self.shape=True; self.paper_architecture=False
        self.conditioning_contract='historical_mmm_shape_dropout'
        self.trans_base=SMPLTransBase(vq,embed_dim,layers,heads,dropout,block_size)
        self.trans_head=CrossCondTransHead(self.num_vq,embed_dim,block_size,layers,heads,dropout,4)

    def forward_function(self,idxs,clip_feature,src_mask=None,att_txt=None,word_emb=None,
                         contact_mask=None,betas=None,gender=None,force_mask=False):
        if contact_mask is not None: raise ValueError('No contact schedule in this experiment')
        if betas is None or gender is None: raise ValueError('Real shape and template condition required')
        mask=self.get_attn_mask(src_mask.repeat_interleave(2,dim=1),att_txt) if src_mask is not None else None
        x=self.trans_base(idxs,clip_feature,mask,word_emb,betas,gender,force_mask)
        logits=self.trans_head(x,mask)[:,1:]
        return logits.reshape(idxs.shape[0],idxs.shape[1],2,self.num_vq)[:,:,[1,0]]

    @torch.no_grad()
    def sample_intra_cas(self,clip_feature,word_emb,betas,gender,m_length,
                         sample_seeds,max_steps=50,temperature=.5,CFG=3.5,
                         confidence_noise_scale=0.):
        device=clip_feature.device; batch=len(clip_feature)
        lengths=torch.ceil(m_length/4).long()
        if torch.any(lengths<1) or torch.any(lengths>self.block_size-1): raise ValueError('Motion length exceeds capacity')
        generators=[torch.Generator(device=device).manual_seed(int(s)%(2**63-1)) for s in sample_seeds]
        mask_id,pad_id,end_id=self.num_vq+2,self.num_vq+1,self.num_vq
        ids=torch.full((batch,self.block_size-1,2),mask_id,dtype=torch.long,device=device)
        valid=generate_src_mask(self.block_size-1,lengths)
        attention=generate_src_mask(self.block_size-1,lengths+1)
        ids[~attention[:,:,None].expand_as(ids)]=pad_id
        for i,n in enumerate(lengths.tolist()):
            if n<self.block_size-1: ids[i,n]=end_id
        for step in range(1,max_steps+1):
            remaining=(cosine_schedule(torch.tensor(step/max_steps,device=device))*lengths).floor().long().clamp_min(0)
            for channel in (1,0):
                masked=(ids[:,:,channel]==mask_id)&valid
                if not masked.any(): continue
                logits=self.forward_function(ids,clip_feature,attention,word_emb=word_emb,betas=betas,gender=gender)
                if CFG>0:
                    uncond=self.forward_function(ids,clip_feature,attention,word_emb=word_emb,betas=betas,gender=gender,force_mask=True)
                    logits=uncond+CFG*(logits-uncond)
                selected=logits[:,:,channel].float()
                proposal=InterHumanText2MotionTransformer._seeded_gumbel(selected,temperature,generators)
                confidence=selected.softmax(-1).gather(-1,proposal[:,:,None]).squeeze(-1)
                uncertainty=(1-confidence).masked_fill(~masked,-float('inf'))
                scale=confidence_noise_scale*(max_steps-step)/max(max_steps-1,1)
                if scale>0:
                    uniforms=torch.stack([torch.rand(confidence[i].shape,device=device,generator=g) for i,g in enumerate(generators)])
                    noise=-(-uniforms.clamp_min(1e-20).log()).clamp_min(1e-20).log()
                    uncertainty=(-(confidence.clamp_min(torch.finfo(confidence.dtype).tiny).log()+scale*noise)).masked_fill(~masked,-float('inf'))
                order=uncertainty.argsort(-1,descending=True)
                keep=torch.zeros_like(masked)
                for i,n in enumerate(remaining.tolist()): keep[i,order[i,:min(n,int(masked[i].sum()))]]=True
                keep &= masked
                ids[:,:,channel][masked&~keep]=proposal[masked&~keep]
        if ((ids==mask_id)&valid[:,:,None]).any(): raise RuntimeError('Unresolved masks')
        return ids
