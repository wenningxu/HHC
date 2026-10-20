"""InterHuman-only HHC transformer extensions."""

from __future__ import annotations

import torch
from torch import nn

from models.two_person_transformer import CrossCondTransBase, CrossCondTransHead, Text2Motion_Transformer
from models.motion_transformer import AdaCrossSelfBlock, Block, Block_crossatt
from models.trans_encdec import PositionEmbeddingSine1D
from utils.runtime import cosine_schedule, generate_src_mask


class InterHumanTransBase(CrossCondTransBase):
    """Two actual parts; reuse HHC attention blocks and joint text dropout."""

    def __init__(self, vqvae, num_vq, embed_dim, clip_dim, block_size,
                 num_layers, num_local_layer, n_head, drop_out_rate, fc_rate,
                 paper_architecture, conditioning_contract):
        nn.Module.__init__(self)
        if len(vqvae.latent_dims) != 2 or len(vqvae.quantizers) != 2:
            raise ValueError('InterHuman MGT requires a body/root two-codebook VQ')
        self.vqvae = vqvae
        self.num_vq = num_vq
        self.block_size = block_size
        self.paper_architecture = paper_architecture
        self.conditioning_contract = conditioning_contract
        self.shape_mask_prob = 0.0 if paper_architecture else 0.2
        self.text_drop_prob = 0.1
        self.special_emb = nn.ModuleList([nn.Embedding(3, d) for d in vqvae.latent_dims])
        self.emb_proj = nn.ModuleList([nn.Linear(d, embed_dim) for d in vqvae.latent_dims])
        self.part_emb = nn.Embedding(2, embed_dim)
        self.pos_embed = PositionEmbeddingSine1D(embed_dim, 0.0, block_size, True)
        self.word_emb = nn.Linear(clip_dim, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        if paper_architecture:
            self.shared_special_emb = nn.Embedding(3, embed_dim)
            self.paper_condition_mlp = nn.Sequential(
                nn.Linear(clip_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
            self.blocks = nn.ModuleList([
                AdaCrossSelfBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers)])
            self.cross_att = nn.ModuleList()
        else:
            self.cross_att = nn.ModuleList([
                Block_crossatt(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_local_layer)])
            self.blocks = nn.ModuleList([
                Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers - num_local_layer)])
        # Do not reinitialize the frozen VQ when initializing MGT modules.
        for name, child in self.named_children():
            if name != 'vqvae':
                child.apply(self._init_weights)

    def forward(self, idxs, clip_feature, src_mask, word_emb, force_mask=False):
        if idxs.ndim != 3 or idxs.shape[-1] != 2 or not 0 < idxs.shape[1] <= self.block_size:
            raise ValueError('InterHuman MGT tokens must be nonempty [B,T,2] within block_size')
        if torch.any(idxs < 0) or torch.any(idxs >= self.num_vq + 3):
            raise ValueError('InterHuman MGT token index out of range')
        parts = []
        for part_id, channel in enumerate((1, 0)):  # interleave root, body
            ids = idxs[..., channel]
            ordinary = ids < self.num_vq
            latent = torch.zeros((*ids.shape, self.vqvae.latent_dims[channel]), device=ids.device)
            if ordinary.any():
                latent[ordinary] = self.vqvae.quantizers[channel].dequantize(ids[ordinary]).detach()
            if (~ordinary).any():
                latent[~ordinary] = self.special_emb[channel](ids[~ordinary] - self.num_vq)
            value = self.emb_proj[channel](latent)
            if self.paper_architecture and (~ordinary).any():
                value[~ordinary] = self.shared_special_emb(ids[~ordinary] - self.num_vq).to(value.dtype)
            parts.append(value + self.pos_embed(value) + self.part_emb.weight[part_id])
        x = torch.stack(parts, dim=2).flatten(1, 2)
        if self.conditioning_contract == 'paper_text_cfg':
            word_emb, clip_feature = self.mask_text(word_emb, clip_feature, force_mask)
        word_emb = self.word_emb(word_emb)
        for block in self.cross_att:
            x = block(x, word_emb)
        if self.paper_architecture:
            condition = self.paper_condition_mlp(clip_feature)
            for block in self.blocks:
                x = block(x, word_emb, condition, src_mask)
        else:
            if self.conditioning_contract == 'historical_mmm_shape_dropout':
                clip_feature = self.mask_shape(clip_feature, force_mask)
            x = torch.cat((self.cond_emb(clip_feature).unsqueeze(1), x), dim=1)
            for block in self.blocks:
                x = block(x, src_mask)
        return x


class InterHumanText2MotionTransformer(Text2Motion_Transformer):
    """Body/root HHC with batch-invariant per-request sampling seeds."""

    def __init__(self, vqvae, num_vq=1024, embed_dim=512, clip_dim=512,
                 block_size=76, num_layers=2, num_local_layer=0, n_head=8,
                 drop_out_rate=0.1, fc_rate=4, contact=False, shape=False,
                 paper_architecture=True, conditioning_contract='paper_text_cfg'):
        nn.Module.__init__(self)
        if contact or shape or conditioning_contract not in {
                'paper_text_cfg', 'historical_mmm_shape_dropout'}:
            raise ValueError('InterHuman MGT requires unshaped supported text conditioning')
        if not 0 <= num_local_layer <= num_layers:
            raise ValueError('num_local_layer must be within [0,num_layers]')
        self.n_head, self.num_vq, self.block_size = n_head, num_vq, block_size
        self.shape = False
        self.paper_architecture = paper_architecture
        self.conditioning_contract = conditioning_contract
        self.trans_base = InterHumanTransBase(
            vqvae, num_vq, embed_dim, clip_dim, block_size, num_layers,
            num_local_layer, n_head, drop_out_rate, fc_rate, paper_architecture,
            conditioning_contract)
        self.trans_head = CrossCondTransHead(
            num_vq, embed_dim, block_size, 0 if paper_architecture else num_layers,
            n_head, drop_out_rate, fc_rate)

    def forward(self, *args, type='forward', **kwargs):
        if type == 'forward':
            return self.forward_function(*args, **kwargs)
        if type == 'sample_intra_cas':
            return self.sample_intra_cas(*args, **kwargs)
        raise ValueError(f'unsupported two-part InterHuman operation: {type}')

    def forward_function(self, idxs, clip_feature, src_mask=None, att_txt=None,
                         word_emb=None, contact_mask=None, betas=None,
                         gender=None, force_mask=False):
        if any(value is not None for value in (contact_mask, betas, gender)):
            raise ValueError('InterHuman MGT accepts no shape/contact conditioning')
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask.repeat_interleave(2, dim=1), att_txt)
        features = self.trans_base(idxs, clip_feature, src_mask, word_emb, force_mask)
        logits = self.trans_head(features, src_mask)
        if not self.paper_architecture:
            logits = logits[:, 1:]
        return logits.reshape(idxs.shape[0], idxs.shape[1], 2, self.num_vq)[:, :, [1, 0]]

    @staticmethod
    def _seeded_gumbel(logits, temperature, generators):
        noises = []
        for batch_index, generator in enumerate(generators):
            uniform = torch.rand(
                logits[batch_index].shape,
                dtype=logits.dtype,
                device=logits.device,
                generator=generator,
            )
            # Match utils.runtime.gumbel_noise exactly; only the RNG stream is
            # made request-local.  In particular, do not clamp the upper tail.
            inner = -torch.log(uniform.clamp(min=1e-20))
            noises.append(-torch.log(inner.clamp(min=1e-20)))
        return (
            logits / max(float(temperature), 1e-10)
            + torch.stack(noises, dim=0)
        ).argmax(dim=-1)

    def sample_intra_cas(
        self, clip_feature, word_emb, contact_mask=None, betas=None,
        gender=None, m_length=None, if_test=False, rand_pos=True, CFG=-1,
        max_steps=3, vocab_contact_map=None, contact_guidance_scale=1.0,
        temperature=0.5, temperature_end=None, sample_seeds=None,
    ):
        if sample_seeds is None:
            sample_seeds = torch.randint(0, 2**63 - 1, (clip_feature.shape[0],))
        if any(value is not None for value in (
            contact_mask, betas, gender, vocab_contact_map,
        )):
            raise ValueError(
                'InterHuman per-sample sampler accepts no Inter-X contact/shape inputs'
            )
        del if_test, contact_guidance_scale
        batch_size = clip_feature.shape[0]
        if int(max_steps) <= 0:
            raise ValueError('InterHuman sampling requires positive max_steps')
        if m_length is None or m_length.shape != (batch_size,):
            raise ValueError('InterHuman sampling requires one motion length per request')
        if torch.any(m_length <= 0) or torch.any(m_length > 4 * (self.block_size - 1)):
            raise ValueError('InterHuman sampling motion lengths exceed token capacity')
        if isinstance(sample_seeds, torch.Tensor):
            seeds = sample_seeds.detach().cpu().reshape(-1).tolist()
        else:
            seeds = list(sample_seeds)
        if len(seeds) != batch_size:
            raise ValueError(
                f'sample_seeds must contain {batch_size} values, got {len(seeds)}'
            )
        device = clip_feature.device
        generators = []
        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) % (2**63 - 1))
            generators.append(generator)

        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        cascade_order = (1, 0)
        ids = torch.full(
            (batch_size, self.block_size - 1, 2), mask_id,
            dtype=torch.long, device=device,
        )
        m_tokens_len = torch.ceil(m_length / 4).long()
        valid_mask = generate_src_mask(self.block_size - 1, m_tokens_len)
        if self.paper_architecture:
            src_token_mask = valid_mask
            ids[~valid_mask.unsqueeze(-1).expand_as(ids)] = pad_id
        else:
            src_token_mask = generate_src_mask(
                self.block_size - 1, m_tokens_len + 1
            )
            ids[~src_token_mask.unsqueeze(-1).expand_as(ids)] = pad_id
            for batch_index, token_len in enumerate(m_tokens_len.tolist()):
                if token_len < self.block_size - 1:
                    ids[batch_index, token_len, :] = end_id

        for step in range(1, int(max_steps) + 1):
            remaining_ratio = cosine_schedule(torch.tensor(
                step / max_steps, device=device, dtype=torch.float32
            ))
            remaining_counts = torch.floor(
                remaining_ratio * m_tokens_len.float()
            ).long().clamp(min=0)
            for channel in cascade_order:
                current = ids[:, :, channel]
                was_masked = (current == mask_id) & valid_mask
                if not was_masked.any():
                    continue
                logits = self.forward_function(
                    ids, clip_feature, src_token_mask, word_emb=word_emb,
                )
                if CFG > 0:
                    unconditional = self.forward_function(
                        ids, clip_feature, src_token_mask,
                        word_emb=word_emb, force_mask=True,
                    )
                    logits = unconditional + CFG * (logits - unconditional)
                channel_logits = logits[:, :, channel, :]
                if rand_pos and temperature_end is not None:
                    fraction = (max_steps - step) / max(max_steps - 1, 1)
                    sample_temperature = (
                        float(temperature_end)
                        + (float(temperature) - float(temperature_end)) * fraction
                    )
                else:
                    sample_temperature = float(temperature) if rand_pos else 0.0
                predictions = self._seeded_gumbel(
                    channel_logits, sample_temperature, generators
                )
                probability = channel_logits.softmax(dim=-1)
                confidence = probability.gather(
                    -1, predictions.unsqueeze(-1)
                ).squeeze(-1)
                uncertainty = (1.0 - confidence).masked_fill(
                    ~was_masked, -float('inf')
                )
                sorted_indices = uncertainty.argsort(dim=-1, descending=True)
                keep_masked = torch.zeros_like(was_masked)
                for batch_index, count in enumerate(remaining_counts.tolist()):
                    count = min(count, int(was_masked[batch_index].sum().item()))
                    if count:
                        keep_masked[
                            batch_index, sorted_indices[batch_index, :count]
                        ] = True
                keep_masked &= was_masked
                updated = current.clone()
                newly_fixed = was_masked & ~keep_masked
                updated[newly_fixed] = predictions[newly_fixed]
                updated[keep_masked] = mask_id
                ids[:, :, channel] = updated
        if ((ids == mask_id) & valid_mask.unsqueeze(-1)).any():
            raise RuntimeError('cascade sampler ended with unresolved mask tokens')
        return ids
