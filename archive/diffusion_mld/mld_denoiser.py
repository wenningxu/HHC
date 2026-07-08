import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from models.trans_encdec import PositionEmbeddingSine1D, TimestepEmbedder
from exit.utils import generate_src_mask, lengths_to_mask

class DenoiserTransformer(nn.Module):
    def __init__(self,
                 h_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 clip_dim=512, history_shape=(2, 56*12), noise_shape=(1, 256), contact_shape=(12, 2), contact=True,
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.clip_dim = clip_dim

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.contact = contact

        # input embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.h_dim)
        self.embed_history = nn.Linear(self.history_shape[-1], self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)

        if contact:
            self.contact_lookup = nn.Embedding(contact_shape[-1], int(self.h_dim/4))
            self.contact_proj = nn.Sequential(
                nn.Linear(contact_shape[0] * int(self.h_dim/4), self.h_dim),
                nn.LayerNorm(self.h_dim),
                nn.SiLU(),
                nn.Linear(self.h_dim, self.h_dim)
            )

        # transformer encoder layers
        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.h_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

        # output projection
        self.output_process = nn.Linear(self.h_dim, self.noise_shape[-1])

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # print('masking cond')
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward(self, x_t, timesteps, y=None):
        """
        x_t: [B, T=1, D]
        timesteps: [batch_size] (int)
        """
        emb_time = self.embed_timestep(timesteps)  # [1, bs, d]
        emb_history = self.embed_history(y['history_motion_normalized']).permute(1, 0, 2)  # [History, bs, d]
        force_mask = y.get('uncond', False)
        emb_text = self.embed_text(self.mask_cond(y['text_embedding'], force_mask=force_mask)).unsqueeze(0)  # [1, bs, d]
        if self.contact:
            contact_emb = self.contact_lookup(y['contact'])  # [B, num_contacts, d/4]
            contact_emb = contact_emb.view(contact_emb.shape[0], -1)  # [B, num_contacts * d/4]
            contact_emb = self.contact_proj(self.mask_cond(contact_emb, force_mask=force_mask)).unsqueeze(0)  # [1, B, d]


        emb_noise = self.embed_noise(x_t).permute(1, 0, 2)  # [1, bs, d]
        # print('emb_time shape:', emb_time.shape, 'emb_text shape:', emb_text.shape, 'emb_history shape:', emb_history.shape, 'emb_noise shape:', emb_noise.shape)
        # print(emb_time.shape, emb_text.shape, emb_history.shape, emb_noise.shape)
        if self.contact:
            xseq = torch.cat((emb_time, emb_text, contact_emb, emb_history, emb_noise), dim=0)
        else:
            xseq = torch.cat((emb_time, emb_text, emb_history, emb_noise), dim=0)
        # print('xseq shape:', xseq.shape)
        xseq = self.sequence_pos_encoder(xseq)

        output = self.seqTransEncoder(xseq)[-self.noise_shape[0]:]  # [1, bs, h_dim]

        output = self.output_process(output)  # [1, B, noise_shape[-1]]
        output = output.permute(1, 0, 2)  # [B, 1, noise_shape[-1]]


        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class DiffusionForcingTransformer(nn.Module):
    def __init__(self,
                 h_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 clip_dim=512, noise_shape=(1, 256), shape_shape=(1, 20), gender_shape=(1, 2), contact_shape=(1, 3),
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.noise_shape = noise_shape
        self.clip_dim = clip_dim

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.contact_mask_prob = kargs.get('contact_mask_prob', 0.)
        # self.contact_drop_id = 2

        # input embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.h_dim)
        # self.embed_shape = nn.Linear(shape_shape[-1], self.h_dim)
        # self.embed_gender = nn.Embedding(num_embeddings=gender_shape[-1], embedding_dim=self.h_dim)
        # self.embed_contact = nn.Embedding(num_embeddings=contact_shape[-1], embedding_dim=self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)



        # transformer encoder layers
        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.h_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

        # output projection
        self.output_process = nn.Linear(self.h_dim, self.noise_shape[-1])

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # print('masking cond')
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x_t, timesteps, y=None):
        """
        x_t: [B, L, D] - 输入序列
        timesteps: [B, L] - 每个 token 独立的噪声水平
        y: dict conditions (keys: 'text_embedding', 'betas', 'gender')
        """
        # 获取 batch 和 length
        B, L, _ = x_t.shape

        # 1. 基础特征编码
        # 将输入噪声投影到隐藏层 [B, L, D] -> [B, L, h]
        # print(f"！！！！！！！！！！！！！！！！！！！！！！！DEBUG: pe shape={self.sequence_pos_encoder.pe.shape}")
        x = self.embed_noise(x_t)
        x = x.permute(1, 0, 2)
        # 2. 时间步编码
        # 对应 Diffusion Forcing 的 Independent Noise Levels
        # 假设 embed_timestep 能处理 [B, L]，如果它输出 [B, L, h] 则直接相加
        time_emb = self.embed_timestep(timesteps)
        # print(timesteps.shape, time_emb.shape, x.shape)
        # x = x + time_emb  # [B, L, h]

        # 3. 条件注入 (关键修改：只对 Text 使用 mask_cond)
        text_emb = self.embed_text(self.mask_cond(y['text_embedding'])).unsqueeze(0)  # [B, h]
        # print(text_emb.shape, x.shape)
        x = torch.cat((time_emb, text_emb, x), dim=0)

        # 4. 序列处理 (Transformer)
        # 维度转换: [B, L, h] -> [L, B, h] (Transformer 标准输入)


        # 位置编码
        x = self.sequence_pos_encoder(x)

        # 生成因果掩码 (Causal Mask)
        # 上三角矩阵 (-inf)，确保位置 i 只能看到 0...i
        # mask = nn.Transformer.generate_square_subsequent_mask(L+1, device=x.device)
        # m_tokens_len = y['m_tokens_len']
        # seq_mask = lengths_to_mask(m_tokens_len + 2, max_len=L+2, device=x.device)  # +1 for text token

        # 编码器前向传播
        # 开启 is_causal=True 激活 Flash Attention 优化
        x = self.seqTransEncoder(x)[2:]

        # 5. 输出投影
        # 维度转换回 [B, L, h]
        x = x.permute(1, 0, 2)
        output = self.output_process(x)
        # print(output.shape)
        return output

class DenoiserTransformerFull(nn.Module):
    def __init__(self,
                 h_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 clip_dim=512, noise_shape=(1, 256),
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.noise_shape = noise_shape
        self.clip_dim = clip_dim

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.contact = kargs.get('contact', False)

        # input embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.time_pos_encoder = PositionalEncoding(int(self.h_dim/4), self.dropout)
        self.embed_timestep = TimestepEmbedder(int(self.h_dim/4), self.time_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)
        if self.contact:
            self.contact_lookup = nn.Embedding(2, int(self.h_dim/4))
            self.init_mlp = nn.Sequential(
                nn.Linear(self.h_dim + int(self.h_dim / 2), self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
            )
        else:

            self.init_mlp = nn.Sequential(
                nn.Linear(self.h_dim + int(self.h_dim/4), self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
            )

        # transformer encoder layers
        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.h_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

        # output projection
        self.output_process = nn.Linear(self.h_dim, self.noise_shape[-1])

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # print('masking cond')
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward(self, x_t, timesteps, y=None, is_causal=True):
        """
        x_t: [B, T=1, D]
        timesteps: [batch_size] (int)
        """
        B, T, D = x_t.shape
        emb_time = self.embed_timestep(timesteps)  # [1, bs, d]
        force_mask = y.get('uncond', False)
        emb_text = self.embed_text(self.mask_cond(y['text_embedding'], force_mask=force_mask)).unsqueeze(0)  # [1, bs, d]
        emb_noise = self.embed_noise(x_t).permute(1, 0, 2)  # [1, bs, d]
        xseq = torch.cat((emb_noise, emb_time), dim=-1)
        if self.contact:
            contact_emb = self.contact_lookup(y['contact'])  # [B, num_contacts, d/4]
            contact_emb = contact_emb.permute(1, 0, 2)  # [num_contacts, B, d/4]
            xseq = torch.cat((xseq, contact_emb), dim=-1)
        xseq = self.init_mlp(xseq)
        xseq = torch.cat((emb_text, xseq), dim=0)
        xseq = self.sequence_pos_encoder(xseq)
        # print('xseq shape:', xseq.shape)
        mask = nn.Transformer.generate_square_subsequent_mask(T+1, xseq.device) if is_causal else None

        output = self.seqTransEncoder(xseq, mask=mask, is_causal=is_causal)[1:]  # [1, bs, h_dim]

        output = self.output_process(output)  # [1, B, noise_shape[-1]]
        output = output.permute(1, 0, 2)  # [B, 1, noise_shape[-1]]


        return output