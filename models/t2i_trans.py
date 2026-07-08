import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from exit.utils import cosine_schedule, gumbel_sample, generate_src_mask
from models.pos_encoding import PositionEmbedding
from models.trans_encdec import PositionEmbeddingSine1D
from models.t2m_trans import Block, Block_crossatt, AdaBlock


class CrossCondTransBase(nn.Module):
    def __init__(self,
                 vqvae,
                 num_vq=1024,
                 embed_dim=512,
                 clip_dim=512,
                 block_size=16,
                 num_layers=2,
                 num_local_layer=1,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4,
                 contact=False,
                 shape=False):
        super().__init__()
        self.vqvae = vqvae
        self.num_vq = num_vq
        self.block_size = block_size
        self.num_local_layer = num_local_layer
        self.total_code_dim = vqvae.code_dim  # VQVAE 的总 latent 维度
        # 计算 VQVAE 中各个部分的维度
        self.upper_dim = int(self.total_code_dim * 3 / 8)
        self.lower_dim = int(self.total_code_dim * 3 / 8)
        self.root_dim = int(self.total_code_dim / 4)

        # 为特殊 Token (End, Pad, Mask) 定义可学习的 Embedding
        # 需要为三个部分分别定义，以便拼接后维度正确
        # [INFO] 3 = [end_id, pad_id, mask_id]
        self.learn_tok_emb_upper = nn.Embedding(3, self.upper_dim)
        self.learn_tok_emb_lower = nn.Embedding(3, self.lower_dim)
        self.learn_tok_emb_root = nn.Embedding(3, self.root_dim)

        # 投影层：将拼接后的 (Upper+Lower+Root) 投影到 Transformer 的 embed_dim
        # self.to_emb = nn.Linear(self.total_code_dim, embed_dim)
        self.part_emb = nn.Embedding(3, embed_dim)
        self.emb_proj_upper = nn.Linear(self.upper_dim, embed_dim)
        self.emb_proj_lower = nn.Linear(self.lower_dim, embed_dim)
        self.emb_proj_root = nn.Linear(self.root_dim, embed_dim)

        self.contact = contact
        if contact:
            self.contact_emb = nn.Embedding(3, embed_dim)
            self.contact_mask_prob = 0.2
            self.contact_drop_id = 2

        self.shape = shape
        if shape:
            gender_emb_dim = 4
            shape_emb_dim = 40
            self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gender_emb_dim)  # 2个类别: 0和1
            self.shape_projection = nn.Linear(20, shape_emb_dim)  # 10是你的体型向量维度
            # 3. 深度融合分支 (MLP)
            combined_input_dim = 2 * gender_emb_dim + shape_emb_dim
            self.condition_mlp = nn.Sequential(
                nn.Linear(combined_input_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.blocks = nn.Sequential(*[AdaBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in
                                          range(num_layers - num_local_layer)])
            self.shape_mask_prob = 0.2
        else:
            self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in
                                          range(num_layers - num_local_layer)])

        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.text_drop_prob = 0.1

        # Transformer Blocks
        self.pos_embed = PositionEmbeddingSine1D(embed_dim, 0.0, block_size, True)

        if num_local_layer > 0:
            self.word_emb = nn.Linear(clip_dim, embed_dim)
            self.cross_att = nn.Sequential(
                *[Block_crossatt(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in
                  range(num_local_layer)])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def mask_cond(self, cond, force_mask=False):
        bs, t = cond.shape
        if force_mask:
            return torch.full_like(cond, self.contact_drop_id)
        elif self.training and self.contact_mask_prob > 0.:
            # print('masking cond')
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.contact_mask_prob).view(bs, 1).bool()  # 1-> use null_cond, 0-> use real cond
            return torch.where(mask,
                                   torch.full_like(cond, self.contact_drop_id),
                                   cond)
        else:
            return cond

    def mask_text(self, word_cond, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(word_cond), torch.zeros_like(cond)
        elif self.training and self.text_drop_prob > 0.:
            # print('masking cond')
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.text_drop_prob).view(bs, 1, 1).bool()  # 1-> use null_cond, 0-> use real cond
            return torch.where(mask,torch.zeros_like(word_cond),word_cond), torch.where(mask,torch.zeros_like(cond),cond)
        else:
            return word_cond, cond

    def mask_shape(self, shape_cond, force_mask=False):
        bs, d = shape_cond.shape
        if force_mask:
            return torch.zeros_like(shape_cond)
        elif self.training and self.shape_mask_prob > 0.:
            # print('masking cond')
            mask = torch.bernoulli(torch.ones(bs, device=shape_cond.device) * self.shape_mask_prob).view(bs, 1).bool()  # 1-> use null_cond, 0-> use real cond
            return torch.where(mask,torch.zeros_like(shape_cond),shape_cond)
        else:
            return shape_cond

    def forward(self, idxs, clip_feature, src_mask, word_emb, contact_mask=None, betas=None, gender=None, force_mask=False):
        # idxs shape: [B, T, 3] -> (Upper, Lower, Root)

        if idxs.shape[1] == 0:  # Handle empty input case if necessary
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t, _ = idxs.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."

            # 分离三个部分的索引
            idx_u = idxs[..., 0]
            idx_l = idxs[..., 1]
            idx_r = idxs[..., 2]

            # 区分普通 Token 和 特殊 Token (Mask/Pad/End)
            # 假设 num_vq 是 Codebook 大小
            not_learn_u = idx_u < self.num_vq
            not_learn_l = idx_l < self.num_vq
            not_learn_r = idx_r < self.num_vq

            # 准备 Embedding 容器
            emb_u = torch.zeros((*idx_u.shape, self.upper_dim), device=idxs.device, dtype=torch.float)
            emb_l = torch.zeros((*idx_l.shape, self.lower_dim), device=idxs.device, dtype=torch.float)
            emb_r = torch.zeros((*idx_r.shape, self.root_dim), device=idxs.device, dtype=torch.float)

            # 1. Upper Body
            # 如果是普通 Token，使用 VQVAE quantizer 解码 (freeze grad)
            if not_learn_u.any():
                emb_u[not_learn_u] = self.vqvae.quantizer_upper.dequantize(idx_u[not_learn_u]).detach()
            # 如果是特殊 Token，使用 learnable embedding (减去 offset)
            if (~not_learn_u).any():
                emb_u[~not_learn_u] = self.learn_tok_emb_upper(idx_u[~not_learn_u] - self.num_vq)

            # 2. Lower Body
            if not_learn_l.any():
                emb_l[not_learn_l] = self.vqvae.quantizer_lower.dequantize(idx_l[not_learn_l]).detach()
            if (~not_learn_l).any():
                emb_l[~not_learn_l] = self.learn_tok_emb_lower(idx_l[~not_learn_l] - self.num_vq)

            # 3. Root
            if not_learn_r.any():
                emb_r[not_learn_r] = self.vqvae.quantizer_root.dequantize(idx_r[not_learn_r]).detach()
            if (~not_learn_r).any():
                emb_r[~not_learn_r] = self.learn_tok_emb_root(idx_r[~not_learn_r] - self.num_vq)

            # [重要] 拼接三个部分，还原成完整的 Latent Vector
            # Shape: [B, T, Total_Code_Dim]
            x_u = self.emb_proj_upper(emb_u)
            x_l = self.emb_proj_lower(emb_l)
            x_r = self.emb_proj_root(emb_r)
            time_pe = self.pos_embed(x_r)


            x_r = x_r + time_pe + self.part_emb(torch.tensor(0, device=idxs.device))
            x_u = x_u + time_pe + self.part_emb(torch.tensor(1, device=idxs.device))
            x_l = x_l + time_pe + self.part_emb(torch.tensor(2, device=idxs.device))

            if (contact_mask is not None) and self.contact:
            # contact_mask shape: [B, T] (0 or 1)
            # c_emb shape: [B, T, D]
                c_emb = self.contact_emb(self.mask_cond(contact_mask, force_mask=force_mask))

                # 广播加法：直接加到三个部位的 Embedding 上
                # 这样每一帧的所有部位都带上了 "是否接触" 的 Context
                x_u = x_u + c_emb
                x_l = x_l + c_emb
                x_r = x_r + c_emb


            x_stack = torch.stack([x_r, x_u, x_l], dim=2)
            B, T, _, D = x_stack.shape
            token_embeddings = x_stack.view(B, T * 3, D)
            # Cross Attention 处理
            if self.num_local_layer > 0:
                word_emb = self.word_emb(word_emb)
                # token_embeddings = self.pos_embed(token_embeddings)
                for module in self.cross_att:
                    token_embeddings = module(token_embeddings, word_emb)

            # 加上 Global Condition (CLIP Feature)
            # 类似于 BERT 的 CLS token，这里作为第一个 token
            clip_feature = self.mask_shape(clip_feature, force_mask=force_mask)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)

        # 进入 Transformer Block
        x = token_embeddings
        if (betas is not None) and self.shape:
            gender_embeded = self.gender_embedding(gender).reshape(-1, 8)
            shape_embeded = self.shape_projection(self.mask_shape(betas, force_mask=False))
            combined = torch.cat([gender_embeded, shape_embeded], dim=1)
            condition_vector = self.condition_mlp(combined)
        for block in self.blocks:
            if isinstance(block, AdaBlock):
                x = block(x, condition_vector, src_mask)
            else:
                x = block(x, src_mask)

        return x


class CrossCondTransHead(nn.Module):
    def __init__(self,
                 num_vq=1024,
                 embed_dim=512,
                 block_size=16,
                 num_layers=2,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(
            *[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_vq, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)

        # 输出三个部分的 Logits
        logits = self.head(x)

        return logits


class Text2Motion_Transformer(nn.Module):
    def __init__(self,
                 vqvae,
                 num_vq=1024,
                 embed_dim=512,
                 clip_dim=512,
                 block_size=16,
                 num_layers=2,
                 num_local_layer=0,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4,
                 contact = False,
                 shape=False):
        super().__init__()
        self.n_head = n_head
        self.num_vq = num_vq
        self.block_size = block_size
        self.shape = shape

        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, clip_dim, block_size, num_layers,
                                             num_local_layer, n_head, drop_out_rate, fc_rate, contact, shape)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)

    def get_attn_mask(self, src_mask, att_txt=None, att_shape=False):
        if att_txt is None:
            att_txt = torch.tensor([[True]] * src_mask.shape[0]).to(src_mask.device)
        # 因为 input 多了一个 condition token，mask 也要加一位
        src_mask = torch.cat([att_txt, src_mask], dim=1)
        if att_shape:
            # 如果有形状条件，也要把它加到 mask 上
            shape_att = torch.tensor([[True]] * src_mask.shape[0]).to(src_mask.device)
            src_mask = torch.cat([shape_att, src_mask], dim=1)

        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        return src_mask

    def forward(self, *args, type='forward', **kwargs):
        if type == 'forward':
            return self.forward_function(*args, **kwargs)
        elif type == 'sample':
            return self.sample(*args, **kwargs)
        elif type == 'sample_cascade':
            return self.sample_cascade(*args, **kwargs)
        elif type == 'sample_intra_cas':
            return self.sample_intra_cas(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')

    def forward_function(self, idxs, clip_feature, src_mask=None, att_txt=None, word_emb=None, contact_mask=None, betas=None, gender=None, force_mask=False):
        # idxs shape: [B, T, 3]
        if src_mask is not None:
            src_mask = src_mask.repeat_interleave(3, dim=1)
            src_mask = self.get_attn_mask(src_mask, att_txt)

        feat = self.trans_base(idxs, clip_feature, src_mask, word_emb, contact_mask, betas, gender, force_mask)
        logits = self.trans_head(feat, src_mask)
        B = logits.shape[0]
        logits = logits[:, 1:, :]
        logits_reshaped = logits.view(B, -1, 3, logits.shape[-1])
        logits_root = logits_reshaped[:, :, 0, :]
        logits_upper = logits_reshaped[:, :, 1, :]
        logits_lower = logits_reshaped[:, :, 2, :]
        # 这里的 logits 是 [B, T+1, 3, Num_VQ]
        # 我们通常不需要 condition token 的预测结果，也不需要它的 logits
        # 但是为了保持形状对齐，通常调用者会处理，或者在这里切片
        # 假设这里返回完整的，或者切掉第一个
        # 去掉 condition token 的预测
        logits_final = torch.stack([logits_upper, logits_lower, logits_root], dim=2)
        return logits_final

    def sample(self, clip_feature, word_emb, contact_mask = None, m_length=None, if_test=False, rand_pos=True, max_steps=10, betas=None, gender=None, CFG=-1):
        # [INFO]
        # m_tokens_len: 有效的 token 长度
        # 3-channel sampling logic mimicking Code 1 (Full Body)
        max_length = 38
        batch_size = clip_feature.shape[0]
        # 特殊 Token ID
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq

        # 初始化 ids: [B, T, 3] 全为 Mask
        shape = (batch_size, self.block_size - 1, 3)
        ids = torch.full(shape, mask_id, dtype=torch.long, device=clip_feature.device)
        scores = torch.ones(shape, dtype=torch.float32, device=clip_feature.device)

        # 计算长度和 Mask
        # 假设 m_length 是帧数，转为 token 数
        m_tokens_len = torch.ceil((m_length) / 4).long()  # 根据你的 VQVAE downsample rate 调整

        # Mask 矩阵 (只针对 T 维度，广播到 3 个 channel)
        # ('block size, m_token_len', self.block_size, m_tokens_len + 1)
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1)  # [B, T]
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len)  # [B, T]

        # 广播 Mask 到 3 个通道: [B, T] -> [B, T, 3]
        src_token_mask_3c = src_token_mask.unsqueeze(-1).expand(-1, -1, 3)
        src_token_mask_noend_3c = src_token_mask_noend.unsqueeze(-1).expand(-1, -1, 3)

        # 填充 Pad 和 End
        # 1. 非 Mask 区域填 Pad
        ids[~src_token_mask_3c] = pad_id

        # 2. 在结束位置填 End
        # gather/scatter 需要 careful indexing.
        # 简单起见，循环 batch 处理 end token，或者使用高级 indexing
        for i in range(batch_size):
            l = m_tokens_len[i]
            if l < self.block_size - 1:
                ids[i, l, :] = end_id

        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8  # 可以根据长度动态调整，这里简化

        for step in range(max_steps):
            # 1. 计算当前 Mask 比例
            timestep = torch.clip(step/(sample_max_steps), max=1)
            if len(m_tokens_len)==1 and step > 0 and torch.clip(step-1/(sample_max_steps), max=1).cpu().item() == timestep:
                break
            rand_mask_prob = cosine_schedule(timestep) # timestep #

            # 计算需要 Mask 掉多少个 Token
            # 注意：这里我们基于 (T * 3) 的总数来计算，或者基于 T 计算然后广播
            # 为了“全身采样”，最好是独立计算每个位置的置信度，然后全局排序

            # [关键修改] 将 (B, T, 3) 展平成 (B, T*3) 进行统一的 MaskGIT 调度
            # 这样网络可以根据置信度决定是保留上半身还是下半身
            flat_len = m_tokens_len * 3
            num_token_masked = (rand_mask_prob * flat_len).long().clip(min=1)

            # 2. 处理无效区域的 Score
            scores[~src_token_mask_noend_3c] = 0  # Pad/End 区域不参与排序
            # 归一化 Score (只在 Valid 区域)
            scores_sum = scores.sum(dim=(1, 2), keepdim=True) + 1e-8
            scores = scores / scores_sum

            token_fix_step = torch.full(shape, max_steps, dtype=torch.long, device=clip_feature.device)
            # 3. 确定哪些位置需要 Mask (基于 Score 排序)
            # Flatten scores to [B, T*3]
            flat_scores = scores.view(batch_size, -1)

            # 加上噪音扰动
            if rand_pos:
                # 这里的实现方式是加入 Gumbel 噪声或者直接 multinomial
                # Code 1 使用 sorted，我们加一点扰动在 scores 上再 sort
                noise = torch.rand_like(flat_scores) * 1e-6
                flat_scores_noisy = flat_scores + noise
                sorted_scores, sorted_indices = flat_scores_noisy.sort(descending=True, dim=-1)
            else:
                sorted_scores, sorted_indices = flat_scores.sort(descending=True, dim=-1)

            # 生成 Mask 索引
            # 取前 num_token_masked 个位置设为 Mask
            # 逻辑：Score 高的(置信度低的)保留为 Mask?
            # MaskGIT 逻辑是：Score 是 "Confidence" (1-prob)。
            # Code 1 logic: scores = 1 - probs. So high score = low probability = needs mask.
            # 是的，Code 1 中 scores = 1 - probs，所以分数越高代表越不自信，越需要被 Mask。

            is_mask_flat = torch.zeros_like(flat_scores, dtype=torch.bool)

            for b in range(batch_size):
                k = num_token_masked[b]
                mask_indices_b = sorted_indices[b, :k]
                is_mask_flat[b, mask_indices_b] = True

            # Reshape back to [B, T, 3]
            is_mask = is_mask_flat.view(shape)

            # 强制保护 Pad 和 End 不被 Mask (虽然 score 设为0了应该没事，但保险起见)
            is_mask = is_mask & src_token_mask_noend_3c

            token_fix_step[is_mask] = max_steps  # 被 Mask 的位置重置为未完成

            # 找到当前这一步被保留(Fixed)，且记录中显示还是"未完成"的位置
            newly_fixed_mask = (~is_mask) & (token_fix_step == max_steps)
            token_fix_step[newly_fixed_mask] = step
            # Apply Mask
            ids = torch.where(is_mask, mask_id, ids)

            # 4. Forward
            # ids: [B, T, 3] -> forward_function -> logits: [B, T+1, 3, V]
            # 这里的 forward_function 内部会处理 idxs 的拼接
            logits = self.forward_function(ids, clip_feature, src_token_mask, word_emb=word_emb,
                                           contact_mask=contact_mask, betas=betas, gender=gender)
            if CFG > 0:
                logits_uncond = self.forward_function(ids, clip_feature, src_token_mask, word_emb=word_emb,
                                                      contact_mask=contact_mask, betas=betas, gender=gender,
                                                      force_mask=True)
                logits = logits_uncond + CFG * (logits - logits_uncond)

            # 去掉第一个 (Condition Token) 的预测
            # logits = logits[:, 1:, :, :]  # [B, T, 3, V]

            # 5. Sampling (Gumbel / Argmax)
            # 温度退火
            if rand_pos:
                temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # 对 logits 进行采样 [B, T, 3]
            pred_ids = gumbel_sample(logits, temperature=temperature, dim=-1)

            # 6. 更新 ids
            # 只更新被 Mask 的位置
            ids = torch.where(is_mask, pred_ids, ids)

            # 7. 更新 Scores (Confidence)
            # 计算预测出的 ID 的概率
            probs = F.softmax(logits, dim=-1)
            # 获取 pred_ids 对应的概率
            # probs: [B, T, 3, V], pred_ids: [B, T, 3]
            pred_probs = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)  # [B, T, 3]

            # MaskGIT Score definition: 1 - probability (Confidence high -> Score low)
            # Wait, Code 1 says: scores = 1 - probs.
            # Then sorting descending puts High Score (Low Prob) at the top.
            # Top K indices are masked. So Low Prob tokens are masked. Correct.
            scores = 1 - pred_probs

            # Mask 掉非 active 的区域
            scores = scores.masked_fill(~is_mask, 0)  # 已经被预测好的(非Mask)下轮得分为0(不参与Mask竞争)
            # 实际上 Code 1 中： ids[~is_mask] 已经固定了，score 应该设为 -inf 还是 0？
            # Code 1 逻辑是每轮重新计算所有位置的 Mask。
            # 这里简化逻辑：我们已经把 ids 更新了。下一轮循环开始时，
            # 我们根据新的全量 scores 重新决定哪些位置变成 mask_id。
            # 所以这里只需要更新 scores 即可。

        return ids

    def sample_cascade(self, clip_feature, word_emb, contact_mask=None, m_length=None, if_test=False, rand_pos=True,
               max_steps=15):
        # [INFO]
        # 强制执行采样顺序: Root -> Body -> Hand
        # 建议 max_steps 设置为 3 的倍数以便均匀分配，例如 12 (4+4+4)

        batch_size = clip_feature.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq

        # =========================================================
        # [配置] 通道映射与顺序 (根据你的模型修改这里)
        # =========================================================
        # 假设: 0=Upper, 1=Lower, 2=Root
        # 你的需求: Root -> Body -> Hand
        # 这里假设 Lower 对应 Hand (或者你需要根据实际情况调整 indices)
        root_ch_idx = 2
        body_ch_idx = 0
        hand_ch_idx = 1

        # 定义阶段顺序
        stage_order = [root_ch_idx, body_ch_idx, hand_ch_idx]
        num_stages = len(stage_order)

        # =========================================================

        # 初始化 ids: [B, T, 3] 全为 Mask
        shape = (batch_size, self.block_size - 1, 3)
        ids = torch.full(shape, mask_id, dtype=torch.long, device=clip_feature.device)
        scores = torch.ones(shape, dtype=torch.float32, device=clip_feature.device)

        # 长度计算
        m_tokens_len = torch.ceil((m_length) / 4).long()

        # Mask 矩阵 (Valid 区域为 True)
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len)
        src_token_mask_noend_3c = src_token_mask_noend.unsqueeze(-1).expand(-1, -1, 3)

        # 填充 Pad
        src_token_mask_3c = src_token_mask.unsqueeze(-1).expand(-1, -1, 3)
        ids[~src_token_mask_3c] = pad_id

        # 填充 End
        for i in range(batch_size):
            l = m_tokens_len[i]
            if l < self.block_size - 1:
                ids[i, l, :] = end_id

        # =========================================================
        # 开始分阶段采样循环
        # =========================================================

        # 我们不再基于 global step 计算 mask ratio，而是基于每个 stage 的 local step

        for step in range(max_steps):

            # 1. 确定当前处于哪个 Stage
            # 简单平均分配步数。例如 max_steps=10: Root(0-3), Body(3-6), Hand(6-9)
            # 你也可以手动指定，例如 [4, 4, 2]
            progress = step / max_steps  # 0.0 -> 1.0
            stage_idx = int(progress * num_stages)
            stage_idx = min(stage_idx, num_stages - 1)  # 防止越界

            # 当前正在活跃的通道 ID
            active_channel = stage_order[stage_idx]

            # 2. 计算当前 Stage 内部的进度 (Local Progress)
            # 把 global progress 映射回 0~1
            stage_start_prog = stage_idx / num_stages
            stage_end_prog = (stage_idx + 1) / num_stages

            # local_t: 当前阶段完成了多少 (0.0 -> 1.0)
            local_t = (progress - stage_start_prog) / (stage_end_prog - stage_start_prog)
            local_t = max(0.0, min(1.0, local_t))

            # 3. 计算 Mask 概率 (Cosine Schedule)
            local_t_tensor = torch.tensor(local_t, device=clip_feature.device, dtype=torch.float32)

            # 3. 计算 Mask 概率 (Cosine Schedule)
            rand_mask_prob = cosine_schedule(local_t_tensor)

            # 4. 构建 Mask (这是核心逻辑)
            # 我们需要对三个通道分别决定 Mask 策略

            is_mask = torch.zeros(shape, dtype=torch.bool, device=clip_feature.device)

            for ch_idx in range(3):  # 遍历 3 个通道

                # 情况 A: 已经完成的 Stage (Past)
                # 策略: 完全不再 Mask，保持固定
                if stage_order.index(ch_idx) < stage_idx:
                    is_mask[:, :, ch_idx] = False

                # 情况 B: 尚未开始的 Stage (Future)
                # 策略: 强制全 Mask
                elif stage_order.index(ch_idx) > stage_idx:
                    is_mask[:, :, ch_idx] = True

                # 情况 C: 当前活跃的 Stage (Active)
                # 策略: 标准 MaskGIT 逻辑，根据 Score 排序保留 Top K
                else:
                    # 计算该通道需要 mask 多少个 token
                    # 这里的 flat_len 只是单通道的长度
                    flat_len_ch = m_tokens_len
                    num_token_masked_ch = (rand_mask_prob * flat_len_ch).long().clip(min=1)

                    # 取出当前通道的 scores: [B, T]
                    ch_scores = scores[:, :, ch_idx].clone()

                    # Mask 掉无效区域 (Pad/End)
                    valid_mask_ch = src_token_mask_noend  # [B, T]
                    ch_scores[~valid_mask_ch] = 100.0  # 设为高分，保证不被 mask (或者 score 定义反了？Check below)
                    # Wait, scores = 1 - probs (Confidence).
                    # High Score = Low Confidence = Should be Masked.
                    # Correct logic:
                    # We sort Descending. Top K gets Masked.
                    # So High Score (Low Conf) -> Top of list -> Masked.
                    # Valid Pad/End should have Score = -1 (High Conf) -> Bottom -> Not Masked.
                    ch_scores[~valid_mask_ch] = -1e9

                    # 加噪
                    if rand_pos:
                        noise = torch.rand_like(ch_scores) * 1e-6
                        ch_scores = ch_scores + noise

                    # 排序 [B, T]
                    _, sorted_indices = ch_scores.sort(descending=True, dim=-1)

                    # 生成该通道的 Mask
                    for b in range(batch_size):
                        k = num_token_masked_ch[b]
                        # Top k unconfident tokens are masked
                        mask_indices_b = sorted_indices[b, :k]
                        is_mask[b, mask_indices_b, ch_idx] = True

            # 保护 Pad 和 End 不被 Mask
            is_mask = is_mask & src_token_mask_noend_3c

            # Apply Mask
            ids = torch.where(is_mask, mask_id, ids)

            # 5. Forward (预测所有通道，但在 Update 时只采纳需要的)
            logits = self.forward_function(ids, clip_feature, src_token_mask, word_emb=word_emb,
                                           contact_mask=contact_mask)

            # 6. Sampling
            if rand_pos:
                temperature = 1  # 或者根据 local_t 退火
            else:
                temperature = 0

            pred_ids = gumbel_sample(logits, temperature=temperature, dim=-1)

            # 7. Update IDs
            ids = torch.where(is_mask, pred_ids, ids)

            # 8. Update Scores
            probs = F.softmax(logits, dim=-1)
            pred_probs = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)

            # Update scores globally (虽然后面只用了 active channel 的)
            scores = 1 - pred_probs
            # scores = scores.masked_fill(~is_mask, 0) # 这行可以不要，保留全量 score 历史信息更好

        return ids

    def sample_intra_cas(self, clip_feature, word_emb, contact_mask=None, betas=None, gender=None, m_length=None, if_test=False, rand_pos=True, CFG=-1,
               max_steps=3, vocab_contact_map=None, contact_guidance_scale=1.0, temperature=None):
        # [INFO]
        # Intra-Step Cascade Sampling
        # 每个 Step 内部顺序：Root -> Body -> Hand
        # 总推理次数 = max_steps * 3

        batch_size = clip_feature.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq

        # =========================================================
        # [配置] 通道顺序
        # =========================================================
        # 假设: 0=Upper(Body), 1=Lower(Hand), 2=Root
        # 你的需求: Root -> Body -> Hand
        root_ch_idx = 2
        body_ch_idx = 0
        hand_ch_idx = 1

        # 定义步内级联顺序
        cascade_order = [root_ch_idx, body_ch_idx, hand_ch_idx]

        # =========================================================

        # 初始化 ids: [B, T, 3] 全为 Mask
        shape = (batch_size, self.block_size - 1, 3)
        ids = torch.full(shape, mask_id, dtype=torch.long, device=clip_feature.device)
        scores = torch.ones(shape, dtype=torch.float32, device=clip_feature.device)
        if vocab_contact_map is not None:
            # [3, V] -> [3, 1, 1, V] 方便后续广播
            vocab_contact_map = vocab_contact_map.to(clip_feature.device).view(3, 1, 1, -1)

            # 确保 contact_mask 是 [B, T] 并且是 float 类型
        if contact_mask is not None:
            contact_mask_tensor = contact_mask.to(clip_feature.device).float()
        else:
            contact_mask_tensor = None

        # 长度计算
        m_tokens_len = torch.ceil((m_length) / 4).long()

        # Mask 矩阵
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len)
        src_token_mask_noend_3c = src_token_mask_noend.unsqueeze(-1).expand(-1, -1, 3)

        # 填充 Pad 和 End
        src_token_mask_3c = src_token_mask.unsqueeze(-1).expand(-1, -1, 3)
        ids[~src_token_mask_3c] = pad_id
        for i in range(batch_size):
            l = m_tokens_len[i]
            if l < self.block_size - 1:
                ids[i, l, :] = end_id

        # =========================================================
        # 主循环
        # =========================================================

        for step in range(max_steps):

            # 1. 计算当前 Step 的 Mask 比例 (全局统一，或者你也可以给不同部位微调)
            # 转为 Tensor 避免之前的报错
            step_tensor = torch.tensor(step, device=clip_feature.device, dtype=torch.float32)
            max_steps_tensor = torch.tensor(max_steps, device=clip_feature.device, dtype=torch.float32)

            progress = (step_tensor + 1) / max_steps_tensor  # +1 确保最后一步是 1.0 (全不mask)
            # cosine schedule: progress 0->1 implies prob 1->0
            rand_mask_prob = cosine_schedule(progress)
            current_guidance_scale = contact_guidance_scale * (1.0 - (step / max_steps))
            # =====================================================
            # [关键] 步内级联循环 (Intra-Step Loop)
            # =====================================================
            scores = []
            for ch_idx in cascade_order:

                # A. Forward
                # 此时的 ids 包含了：
                # - 之前 steps 确定的 token
                # - 本 step 刚刚更新过的 前置通道 token (例如预测 Body 时，Root 已经是新的了)

                logits = self.forward_function(ids, clip_feature, src_token_mask, word_emb=word_emb,
                                               contact_mask=contact_mask, betas=betas, gender=gender)
                if CFG > 0:
                    logits_uncond = self.forward_function(ids, clip_feature, src_token_mask, word_emb=word_emb,
                                                   contact_mask=contact_mask, betas=betas, gender=gender, force_mask=True)
                    logits = logits_uncond + CFG * (logits - logits_uncond)
                # B. 获取当前通道的 Logits 和 Probs
                # logits: [B, T, 3, V] -> 取出当前 ch_idx
                logits_ch = logits[:, :, ch_idx, :]  # [B, T, V]
                if vocab_contact_map is not None and contact_mask_tensor is not None and current_guidance_scale > 0.2:
                    # 1. 获取当前通道对应的 Codebook 属性 [1, 1, V]
                    # ch_idx 0=Body, 1=Hand, 2=Root. 你的 npy 也是这个顺序 (3, V)
                    vocab_attr = vocab_contact_map[ch_idx]

                    # 2. 准备目标 Target [B, T, 1]
                    # contact_mask_tensor: 1.0 表示要求接触, 0.0 表示要求分开
                    target = contact_mask_tensor.unsqueeze(-1)

                    # 3. 计算“兼容性分数” (Compatibility Score)
                    # 逻辑：
                    # 如果 Target=1 (接触): 我们希望 vocab_attr (接触概率) 越大越好 -> Score = vocab_attr
                    # 如果 Target=0 (分开): 我们希望 vocab_attr (接触概率) 越小越好 -> Score = 1 - vocab_attr
                    # 公式合并:
                    compatibility = target * vocab_attr + (1.0 - target) * (1.0 - vocab_attr)

                    # 4. 转换为 Logit Bias
                    # 加一个 epsilon 防止 log(0)
                    # Logits += Scale * log(P_compatible)
                    # 兼容性越高，Bias 越大 (接近0)；兼容性越低，Bias 越小 (负无穷)
                    bias = torch.log(compatibility + 1e-6) * current_guidance_scale

                    # 5. 应用到 Logits
                    logits_ch = logits_ch + bias

                # C. 采样 (Gumbel / Argmax)
                if rand_pos:
                    current_temperature = 1.0 if temperature is None else float(temperature)
                else:
                    current_temperature = 0.

                # [B, T]
                pred_ids_ch = gumbel_sample(logits_ch, temperature=current_temperature, dim=-1)

                # D. 计算当前通道的 Confidence (用于 MaskGIT 筛选)
                probs_ch = F.softmax(logits_ch, dim=-1)  # [B, T, V]
                token_probs_ch = probs_ch.gather(-1, pred_ids_ch.unsqueeze(-1)).squeeze(-1)  # [B, T]

                # MaskGIT score: 1 - prob (越高越容易被 mask)
                scores_ch = 1 - token_probs_ch

                # 处理 Pad/End: 设为极小值(极高置信度)，保证不被 Mask
                valid_mask_ch = src_token_mask_noend  # [B, T]
                scores_ch[~valid_mask_ch] = -1e9
                # 加噪
                # if rand_pos:
                #     noise = torch.rand_like(scores_ch) * 1e-6
                #     scores_ch = scores_ch + noise

                # E. 决定哪些位置需要 Mask
                # 计算需要 mask 的数量 (基于当前 step 的全局进度)
                flat_len_ch = m_tokens_len
                num_token_masked_ch = (rand_mask_prob * flat_len_ch).long().clip(min=0)  # 允许为0

                # 排序
                _, sorted_indices = scores_ch.sort(descending=True, dim=-1)

                # 生成 Mask
                is_mask_ch = torch.zeros_like(scores_ch, dtype=torch.bool)
                for b in range(batch_size):
                    k = num_token_masked_ch[b]
                    if k > 0:
                        mask_indices = sorted_indices[b, :k]
                        is_mask_ch[b, mask_indices] = True

                # 保护 Pad
                is_mask_ch = is_mask_ch & valid_mask_ch

                # F. 立即更新 ids 中的当前通道
                # 这样下一次循环 (下一个 ch_idx) 就能看到这里更新后的 token 了

                # 先把当前通道全部更新为预测值
                # (注意：MaskGIT 的标准做法是保留高置信度的，Mask 低置信度的)
                # 这里的逻辑是：
                # 1. 凡是 is_mask=True 的地方，填入 mask_id
                # 2. 凡是 is_mask=False 的地方，填入 pred_ids_ch (即保留预测结果)

                # 取出原来的 ids 通道
                current_ids_ch = ids[:, :, ch_idx]

                # 更新逻辑：
                # 如果被 mask -> mask_id
                # 如果没被 mask -> 使用刚刚预测的 pred_ids_ch (不管它原来是啥，因为我们相信新的预测更好)
                new_ids_ch = torch.where(is_mask_ch, mask_id, pred_ids_ch)

                # 写入回 ids
                ids[:, :, ch_idx] = new_ids_ch

        return ids


class Chi3DCondTransBase(nn.Module):
    def __init__(self,
                 vqvae,
                 num_actions,
                 num_groups=3,
                 num_vq=1024,
                 embed_dim=512,
                 block_size=16,
                 num_layers=2,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4):
        super().__init__()
        self.vqvae = vqvae
        self.num_vq = num_vq
        self.block_size = block_size
        self.total_code_dim = vqvae.code_dim
        self.upper_dim = int(self.total_code_dim * 3 / 8)
        self.lower_dim = int(self.total_code_dim * 3 / 8)
        self.root_dim = int(self.total_code_dim / 4)
        self.group_offset = 2

        self.learn_tok_emb_upper = nn.Embedding(3, self.upper_dim)
        self.learn_tok_emb_lower = nn.Embedding(3, self.lower_dim)
        self.learn_tok_emb_root = nn.Embedding(3, self.root_dim)

        self.part_emb = nn.Embedding(3, embed_dim)
        self.emb_proj_upper = nn.Linear(self.upper_dim, embed_dim)
        self.emb_proj_lower = nn.Linear(self.lower_dim, embed_dim)
        self.emb_proj_root = nn.Linear(self.root_dim, embed_dim)
        self.pos_embed = PositionEmbeddingSine1D(embed_dim, 0.0, block_size, True)

        self.action_embedding = nn.Embedding(num_actions, embed_dim)
        self.group_embedding = nn.Embedding(num_groups, embed_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cond_drop_prob = 0.1
        self.blocks = nn.Sequential(*[
            AdaBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
            for _ in range(num_layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def mask_cond(self, cond, force_mask=False):
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(cond.shape[0], device=cond.device) * self.cond_drop_prob
            ).view(-1, 1).bool()
            return torch.where(mask, torch.zeros_like(cond), cond)
        return cond

    def get_condition_vector(self, action_ids, group_ids, force_mask=False):
        if group_ids.min() >= self.group_offset:
            group_ids = group_ids - self.group_offset
        action_cond = self.action_embedding(action_ids.long())
        group_cond = self.group_embedding(group_ids.long())
        cond = torch.cat([action_cond, group_cond], dim=-1)
        cond = self.mask_cond(cond, force_mask=force_mask)
        return self.condition_mlp(cond)

    def forward(self, idxs, action_ids, group_ids, src_mask=None, force_mask=False):
        b, t, _ = idxs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        idx_u = idxs[..., 0]
        idx_l = idxs[..., 1]
        idx_r = idxs[..., 2]

        not_learn_u = idx_u < self.num_vq
        not_learn_l = idx_l < self.num_vq
        not_learn_r = idx_r < self.num_vq

        emb_u = torch.zeros((*idx_u.shape, self.upper_dim), device=idxs.device, dtype=torch.float)
        emb_l = torch.zeros((*idx_l.shape, self.lower_dim), device=idxs.device, dtype=torch.float)
        emb_r = torch.zeros((*idx_r.shape, self.root_dim), device=idxs.device, dtype=torch.float)

        if not_learn_u.any():
            emb_u[not_learn_u] = self.vqvae.quantizer_upper.dequantize(idx_u[not_learn_u]).detach()
        if (~not_learn_u).any():
            emb_u[~not_learn_u] = self.learn_tok_emb_upper(idx_u[~not_learn_u] - self.num_vq)

        if not_learn_l.any():
            emb_l[not_learn_l] = self.vqvae.quantizer_lower.dequantize(idx_l[not_learn_l]).detach()
        if (~not_learn_l).any():
            emb_l[~not_learn_l] = self.learn_tok_emb_lower(idx_l[~not_learn_l] - self.num_vq)

        if not_learn_r.any():
            emb_r[not_learn_r] = self.vqvae.quantizer_root.dequantize(idx_r[not_learn_r]).detach()
        if (~not_learn_r).any():
            emb_r[~not_learn_r] = self.learn_tok_emb_root(idx_r[~not_learn_r] - self.num_vq)

        x_u = self.emb_proj_upper(emb_u)
        x_l = self.emb_proj_lower(emb_l)
        x_r = self.emb_proj_root(emb_r)
        time_pe = self.pos_embed(x_r)

        x_r = x_r + time_pe + self.part_emb(torch.tensor(0, device=idxs.device))
        x_u = x_u + time_pe + self.part_emb(torch.tensor(1, device=idxs.device))
        x_l = x_l + time_pe + self.part_emb(torch.tensor(2, device=idxs.device))

        x_stack = torch.stack([x_r, x_u, x_l], dim=2)
        x = x_stack.view(b, t * 3, -1)
        cond_vec = self.get_condition_vector(action_ids, group_ids, force_mask=force_mask)

        for block in self.blocks:
            x = block(x, cond_vec, src_mask)
        return x


class Chi3DActionTransformer(nn.Module):
    def __init__(self,
                 vqvae,
                 num_actions,
                 num_groups=3,
                 num_vq=1024,
                 embed_dim=512,
                 block_size=16,
                 num_layers=2,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4):
        super().__init__()
        self.n_head = n_head
        self.num_vq = num_vq
        self.block_size = block_size
        self.trans_base = Chi3DCondTransBase(
            vqvae=vqvae,
            num_actions=num_actions,
            num_groups=num_groups,
            num_vq=num_vq,
            embed_dim=embed_dim,
            block_size=block_size,
            num_layers=num_layers,
            n_head=n_head,
            drop_out_rate=drop_out_rate,
            fc_rate=fc_rate,
        )
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)

    def get_attn_mask(self, src_mask):
        b, t = src_mask.shape
        return src_mask.view(b, 1, 1, t).repeat(1, self.n_head, t, 1)

    def _init_sampling_ids(self, cond_ids, m_length):
        batch_size = cond_ids.shape[0]
        max_length = self.block_size
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq

        shape = (batch_size, max_length, 3)
        ids = torch.full(shape, mask_id, dtype=torch.long, device=cond_ids.device)
        scores = torch.ones(shape, dtype=torch.float32, device=cond_ids.device)
        m_tokens_len = torch.ceil(m_length / 4).long()

        src_token_mask = generate_src_mask(max_length, m_tokens_len + 1)
        src_token_mask_noend = generate_src_mask(max_length, m_tokens_len)
        src_token_mask_3c = src_token_mask.unsqueeze(-1).expand(-1, -1, 3)
        src_token_mask_noend_3c = src_token_mask_noend.unsqueeze(-1).expand(-1, -1, 3)

        ids[~src_token_mask_3c] = pad_id
        for i in range(batch_size):
            token_len = m_tokens_len[i]
            if token_len < max_length:
                ids[i, token_len, :] = end_id

        return ids, scores, m_tokens_len, src_token_mask, src_token_mask_noend, src_token_mask_noend_3c, shape

    def forward(self, idxs, action_ids, group_ids, src_mask=None, force_mask=False):
        if src_mask is not None:
            src_mask = src_mask.repeat_interleave(3, dim=1)
            src_mask = self.get_attn_mask(src_mask)
        feat = self.trans_base(idxs, action_ids, group_ids, src_mask=src_mask, force_mask=force_mask)
        logits = self.trans_head(feat, src_mask)
        b = logits.shape[0]
        logits_reshaped = logits.view(b, -1, 3, logits.shape[-1])
        logits_root = logits_reshaped[:, :, 0, :]
        logits_upper = logits_reshaped[:, :, 1, :]
        logits_lower = logits_reshaped[:, :, 2, :]
        return torch.stack([logits_upper, logits_lower, logits_root], dim=2)

    def sample(self, action_ids, group_ids, m_length=None, if_test=False, rand_pos=True, max_steps=10, CFG=-1):
        ids, scores, m_tokens_len, src_token_mask, _, src_token_mask_noend_3c, shape = self._init_sampling_ids(
            action_ids, m_length
        )
        sample_max_steps = torch.round(max_steps / self.block_size * m_tokens_len) + 1e-8
        mask_id = self.num_vq + 2
        batch_size = action_ids.shape[0]
        for step in range(max_steps):
            timestep = torch.clip(step / sample_max_steps, max=1)
            rand_mask_prob = cosine_schedule(timestep)
            flat_len = m_tokens_len * 3
            num_token_masked = (rand_mask_prob * flat_len).long().clip(min=1)
            scores[~src_token_mask_noend_3c] = 0
            scores_sum = scores.sum(dim=(1, 2), keepdim=True) + 1e-8
            scores = scores / scores_sum
            flat_scores = scores.view(batch_size, -1)
            if rand_pos:
                noise = torch.rand_like(flat_scores) * 1e-6
                _, sorted_indices = (flat_scores + noise).sort(descending=True, dim=-1)
            else:
                _, sorted_indices = flat_scores.sort(descending=True, dim=-1)

            is_mask_flat = torch.zeros_like(flat_scores, dtype=torch.bool)
            for b in range(batch_size):
                k = num_token_masked[b]
                is_mask_flat[b, sorted_indices[b, :k]] = True
            is_mask = is_mask_flat.view(shape) & src_token_mask_noend_3c
            ids = torch.where(is_mask, mask_id, ids)
            logits = self.forward(ids, action_ids, group_ids, src_mask=src_token_mask)
            if CFG > 0:
                logits_uncond = self.forward(ids, action_ids, group_ids, src_mask=src_token_mask, force_mask=True)
                logits = logits_uncond + CFG * (logits - logits_uncond)
            temperature = 1 if rand_pos else 0
            pred_ids = gumbel_sample(logits, temperature=temperature, dim=-1)
            ids = torch.where(is_mask, pred_ids, ids)
            probs = F.softmax(logits, dim=-1)
            pred_probs = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)
            scores = (1 - pred_probs).masked_fill(~is_mask, 0)
        return ids

    def sample_cascade(self, action_ids, group_ids, m_length=None, if_test=False, rand_pos=True,
               max_steps=15, CFG=-1):
        # [INFO]
        # 强制执行采样顺序: Root -> Body -> Hand
        # 建议 max_steps 设置为 3 的倍数以便均匀分配，例如 12 (4+4+4)
        root_ch_idx = 2
        body_ch_idx = 0
        hand_ch_idx = 1
        stage_order = [root_ch_idx, body_ch_idx, hand_ch_idx]
        num_stages = len(stage_order)
        ids, scores, m_tokens_len, src_token_mask, src_token_mask_noend, src_token_mask_noend_3c, shape = (
            self._init_sampling_ids(action_ids, m_length)
        )
        mask_id = self.num_vq + 2
        batch_size = action_ids.shape[0]

        for step in range(max_steps):
            progress = step / max_steps  # 0.0 -> 1.0
            stage_idx = int(progress * num_stages)
            stage_idx = min(stage_idx, num_stages - 1)  # 防止越界
            stage_start_prog = stage_idx / num_stages
            stage_end_prog = (stage_idx + 1) / num_stages
            local_t = (progress - stage_start_prog) / (stage_end_prog - stage_start_prog)
            local_t = max(0.0, min(1.0, local_t))
            local_t_tensor = torch.tensor(local_t, device=action_ids.device, dtype=torch.float32)
            rand_mask_prob = cosine_schedule(local_t_tensor)
            is_mask = torch.zeros(shape, dtype=torch.bool, device=action_ids.device)

            for ch_idx in range(3):  # 遍历 3 个通道
                if stage_order.index(ch_idx) < stage_idx:
                    is_mask[:, :, ch_idx] = False
                elif stage_order.index(ch_idx) > stage_idx:
                    is_mask[:, :, ch_idx] = True
                else:
                    flat_len_ch = m_tokens_len
                    num_token_masked_ch = (rand_mask_prob * flat_len_ch).long().clip(min=1)
                    ch_scores = scores[:, :, ch_idx].clone()
                    valid_mask_ch = src_token_mask_noend  # [B, T]
                    ch_scores[~valid_mask_ch] = -1e9
                    if rand_pos:
                        noise = torch.rand_like(ch_scores) * 1e-6
                        ch_scores = ch_scores + noise
                    _, sorted_indices = ch_scores.sort(descending=True, dim=-1)
                    for b in range(batch_size):
                        k = num_token_masked_ch[b]
                        mask_indices_b = sorted_indices[b, :k]
                        is_mask[b, mask_indices_b, ch_idx] = True

            is_mask = is_mask & src_token_mask_noend_3c
            ids = torch.where(is_mask, mask_id, ids)
            logits = self.forward(ids, action_ids, group_ids, src_mask=src_token_mask)
            if CFG > 0:
                logits_uncond = self.forward(ids, action_ids, group_ids, src_mask=src_token_mask, force_mask=True)
                logits = logits_uncond + CFG * (logits - logits_uncond)
            if rand_pos:
                temperature = 1
            else:
                temperature = 0

            pred_ids = gumbel_sample(logits, temperature=temperature, dim=-1)
            ids = torch.where(is_mask, pred_ids, ids)
            probs = F.softmax(logits, dim=-1)
            pred_probs = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)
            scores = 1 - pred_probs

        return ids

    def sample_intra_cas(self, action_ids, group_ids, m_length=None, if_test=False, rand_pos=True, CFG=-1,
               max_steps=3, temperature=None):
        mask_id = self.num_vq + 2
        cascade_order = [2, 0, 1]
        ids, _, m_tokens_len, src_token_mask, src_token_mask_noend, _, _ = self._init_sampling_ids(
            action_ids, m_length
        )
        batch_size = action_ids.shape[0]

        for step in range(max_steps):
            step_tensor = torch.tensor(step, device=action_ids.device, dtype=torch.float32)
            max_steps_tensor = torch.tensor(max_steps, device=action_ids.device, dtype=torch.float32)

            progress = (step_tensor + 1) / max_steps_tensor
            rand_mask_prob = cosine_schedule(progress)
            for ch_idx in cascade_order:
                logits = self.forward(ids, action_ids, group_ids, src_mask=src_token_mask)
                if CFG > 0:
                    logits_uncond = self.forward(ids, action_ids, group_ids, src_mask=src_token_mask, force_mask=True)
                    logits = logits_uncond + CFG * (logits - logits_uncond)
                logits_ch = logits[:, :, ch_idx, :]

                if rand_pos:
                    current_temperature = 1.0 if temperature is None else float(temperature)
                else:
                    current_temperature = 0.

                pred_ids_ch = gumbel_sample(logits_ch, temperature=current_temperature, dim=-1)
                probs_ch = F.softmax(logits_ch, dim=-1)
                token_probs_ch = probs_ch.gather(-1, pred_ids_ch.unsqueeze(-1)).squeeze(-1)
                scores_ch = 1 - token_probs_ch
                valid_mask_ch = src_token_mask_noend
                scores_ch[~valid_mask_ch] = -1e9

                flat_len_ch = m_tokens_len
                num_token_masked_ch = (rand_mask_prob * flat_len_ch).long().clip(min=0)
                _, sorted_indices = scores_ch.sort(descending=True, dim=-1)

                is_mask_ch = torch.zeros_like(scores_ch, dtype=torch.bool)
                for b in range(batch_size):
                    k = num_token_masked_ch[b]
                    if k > 0:
                        mask_indices = sorted_indices[b, :k]
                        is_mask_ch[b, mask_indices] = True

                is_mask_ch = is_mask_ch & valid_mask_ch
                new_ids_ch = torch.where(is_mask_ch, mask_id, pred_ids_ch)
                ids[:, :, ch_idx] = new_ids_ch

        return ids
