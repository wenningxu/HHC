import torch
import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset
from utils.motion_masks import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK, HHI_BODY_MASK, HHI_HAND_MASK, HHI_ROOT_MASK, HHI_ROOT_MASK_OLD
from utils.fk import FK_Pair_Layer_Batch
from utils.paramUtil import SMPLX_TREE, INTERX_SKELETONS
from models.chi3d_utils import load_chi3d_group_skeletons

class VQVAE_SEP3_shape(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 moment=None,
                 sep_decoder=False):
        super().__init__()
        if args.dataname == 'kit':
            self.nb_joints = 21
            output_dim = 251
            upper_dim = 120
            lower_dim = 131
        elif args.dataname == 'hhi' or args.dataname == 'hhi_shape':
            self.nb_joints = 55
            upper_dim = 22 * 12
            lower_dim = 30 * 12
            root_dim = 12 if args.motion_rep == 'vel' else 6
            output_dim = upper_dim + lower_dim + root_dim
            self.upper_mask = HHI_BODY_MASK
            self.lower_mask = HHI_HAND_MASK
            self.root_mask = HHI_ROOT_MASK if args.motion_rep == 'vel' else HHI_ROOT_MASK_OLD
        else:
            self.nb_joints = 22
            output_dim = 263
            upper_dim = 156
            lower_dim = 107
            self.upper_mask = HML_UPPER_BODY_MASK
            self.lower_mask = HML_LOWER_BODY_MASK
        self.code_dim = code_dim

        gender_emb_dim = 4
        shape_emb_dim = 40
        self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gender_emb_dim)  # 2个类别: 0和1
        self.shape_projection = nn.Linear(20, shape_emb_dim)  # 10是你的体型向量维度
        # 3. 深度融合分支 (MLP)
        combined_input_dim = 2 * gender_emb_dim + shape_emb_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, int(code_dim/32))
        )
        self.dropout = nn.Dropout(0.1)

        self.sep_decoder = sep_decoder
        if self.sep_decoder:
            self.decoder_upper = Decoder(upper_dim, int(code_dim / 2), down_t, stride_t, width, depth,
                                         dilation_growth_rate, activation=activation, norm=norm)
            self.decoder_lower = Decoder(lower_dim, int(code_dim / 2), down_t, stride_t, width, depth,
                                         dilation_growth_rate, activation=activation, norm=norm)
        else:

            self.decoder = Decoder(output_dim, code_dim+int(code_dim/32), down_t, stride_t, width, depth, dilation_growth_rate,
                                   activation=activation, norm=norm)

        self.num_code = nb_code

        self.encoder_upper = Encoder(upper_dim, int(code_dim * 3 / 8), down_t, stride_t, width, depth, dilation_growth_rate,
                                     activation=activation, norm=norm)
        self.encoder_lower = Encoder(lower_dim, int(code_dim * 3 / 8), down_t, stride_t, width, depth, dilation_growth_rate,
                                     activation=activation, norm=norm)
        self.encoder_root = Encoder(root_dim, int(code_dim / 4), down_t, stride_t, width, depth, dilation_growth_rate,
                                     activation=activation, norm=norm)



        self.quantizer_upper = QuantizeEMAReset(nb_code, int(code_dim * 3 / 8), args)
        self.quantizer_lower = QuantizeEMAReset(nb_code, int(code_dim * 3 / 8), args)
        self.quantizer_root = QuantizeEMAReset(nb_code, int(code_dim / 4), args)

        self.shape_mask_prob = 0.1
        interx_skeleton = torch.from_numpy(INTERX_SKELETONS)
        self.fk_layers = FK_Pair_Layer_Batch(
            SMPLX_TREE,
            interx_skeleton[:, 0],
            interx_skeleton[:, 1],
            args.device
        )

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

    def forward(self, x, beta, gender, type='full'):
        '''type=[full, encode, decode]'''
        if type == 'full':
            x = x.float()
            # x = self.shift_upper_down(x)

            upper_emb = x[..., self.upper_mask]
            lower_emb = x[..., self.lower_mask]
            root_emb = x[..., self.root_mask]
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb, loss_upper, perplexity_u = self.quantizer_upper(upper_emb)

            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb, loss_lower, perplexity_l = self.quantizer_lower(lower_emb)
            loss = loss_upper + loss_lower

            root_emb = self.preprocess(root_emb)
            root_emb = self.encoder_root(root_emb)
            root_emb, loss_root, perplexity_r = self.quantizer_root(root_emb)
            loss += loss_root

            perplexity = torch.cat((perplexity_u.unsqueeze(0), perplexity_r.unsqueeze(0), perplexity_l.unsqueeze(0)))
            ## decoder
            if self.sep_decoder:
                x_decoder_upper = self.decoder_upper(upper_emb)
                x_decoder_upper = self.postprocess(x_decoder_upper)
                x_decoder_lower = self.decoder_lower(lower_emb)
                x_decoder_lower = self.postprocess(x_decoder_lower)
                x_out = merge_upper_lower(x_decoder_upper, x_decoder_lower)
                x_out = self.shift_upper_up(x_out)

            else:
                u_in, l_in, r_in = upper_emb, lower_emb, root_emb

                # 仅在训练时触发
                # if self.training and self.part_drop_prob > 0:
                #     # 生成一个 0~1 的随机数，决定是否触发本轮 dropout
                #     if torch.rand(1).item() < self.part_drop_prob:
                #         # 随机选一个倒霉蛋 (0:Upper, 1:Lower, 2:Root)
                #         drop_idx = torch.randint(0, 3, (1,)).item()
                #
                #         if drop_idx == 0:
                #             # 将 Upper 置为全 0 (Mask Out)
                #             u_in = torch.zeros_like(upper_emb)
                #         elif drop_idx == 1:
                #             l_in = torch.zeros_like(lower_emb)
                #         elif drop_idx == 2:
                #             r_in = torch.zeros_like(root_emb)

                # 拼接处理后的输入
                x_quantized = torch.cat([u_in, l_in, r_in], dim=1)
                # print(x_quantized.shape)
                # x_quantized = self.dropout(x_quantized)
                beta = self.mask_shape(beta)
                gender_embeded = self.gender_embedding(gender).reshape(-1, 8)
                shape_embeded = self.shape_projection(beta)
                combined = torch.cat([gender_embeded, shape_embeded], dim=1)
                condition_vector = self.condition_mlp(combined)
                condition_vector = condition_vector.unsqueeze(-1).expand(-1, -1,
                                                                         x_quantized.shape[-1])  # ([256, 32, 16])
                x_quantized = torch.cat([x_quantized, condition_vector], dim=1)

                x_decoder = self.decoder(x_quantized)
                x_out = self.postprocess(x_decoder)
                # print('x_out', x_out.shape)
                x_out = self.merge_body_hand(x_out[..., :22 * 12], x_out[..., 22 * 12:52 * 12], x_out[..., 52 * 12:])

            return x_out, loss, perplexity
        elif type == 'encode':
            N, T, _ = x.shape
            upper_emb = x[..., self.upper_mask]
            lower_emb = x[..., self.lower_mask]
            root_emb = x[..., self.root_mask]

            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb = self.postprocess(upper_emb)
            upper_emb = upper_emb.reshape(-1, upper_emb.shape[-1])
            upper_code_idx = self.quantizer_upper.quantize(upper_emb)
            upper_code_idx = upper_code_idx.view(N, -1)

            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb = self.postprocess(lower_emb)
            lower_emb = lower_emb.reshape(-1, lower_emb.shape[-1])
            lower_code_idx = self.quantizer_lower.quantize(lower_emb)
            lower_code_idx = lower_code_idx.view(N, -1)

            root_emb = self.preprocess(root_emb)
            root_emb = self.encoder_root(root_emb)
            root_emb = self.postprocess(root_emb)
            root_emb = root_emb.reshape(-1, root_emb.shape[-1])
            root_code_idx = self.quantizer_root.quantize(root_emb)
            root_code_idx = root_code_idx.view(N, -1)

            code_idx = torch.stack([upper_code_idx, lower_code_idx, root_code_idx], dim=-1)
            return code_idx

        elif type == 'decode':
            if self.sep_decoder:
                pass  # 如果需要支持 sep_decoder，这里也要相应修改
            else:
                x_d_upper = self.quantizer_upper.dequantize(x[..., 0])
                x_d_lower = self.quantizer_lower.dequantize(x[..., 1])
                x_d_root = self.quantizer_root.dequantize(x[..., 2])

                x_d = torch.cat([x_d_upper, x_d_lower, x_d_root], dim=-1)  # (N, T, C_upper+C_lower+C_root)
                x_d = x_d.permute(0, 2, 1).contiguous()  # (N, C_total, T)

                # Condition
                gender_embeded = self.gender_embedding(gender).reshape(-1, 8)
                shape_embeded = self.shape_projection(beta)
                combined = torch.cat([gender_embeded, shape_embeded], dim=1)
                condition_vector = self.condition_mlp(combined)

                # Expand condition
                condition_vector = condition_vector.unsqueeze(-1).expand(-1, -1, x_d.shape[-1])

                # Concat condition
                x_d = torch.cat([x_d, condition_vector], dim=1)

                # Decoder
                x_decoder = self.decoder(x_d)
                x_out = self.postprocess(x_decoder)
                x_out = self.merge_body_hand(x_out[..., :22 * 12], x_out[..., 22 * 12:52 * 12], x_out[..., 52 * 12:])
                return x_out

    def merge_body_hand(self, body, hand, root):

        motion = torch.zeros(*body.shape[:2], 56 * 12).to(body.device)
        motion[..., self.upper_mask] = body
        motion[..., self.lower_mask] = hand
        motion[..., self.root_mask] = root
        return motion

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x


class VQVAE_SEP3_Chi3DShape(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 moment=None,
                 sep_decoder=False):
        super().__init__()
        self.nb_joints = 55
        upper_dim = 22 * 12
        lower_dim = 30 * 12
        root_dim = 12 if args.motion_rep == 'vel' else 6
        output_dim = upper_dim + lower_dim + root_dim
        self.upper_mask = HHI_BODY_MASK
        self.lower_mask = HHI_HAND_MASK
        self.root_mask = HHI_ROOT_MASK if args.motion_rep == 'vel' else HHI_ROOT_MASK_OLD
        self.code_dim = code_dim
        self.num_code = nb_code
        self.sep_decoder = sep_decoder
        self.group_mask_prob = 0.1
        self.group_offset = 2

        group_emb_dim = 8
        self.group_embedding = nn.Embedding(3, group_emb_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(group_emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, int(code_dim / 32))
        )

        if self.sep_decoder:
            self.decoder_upper = Decoder(upper_dim, int(code_dim / 2), down_t, stride_t, width, depth,
                                         dilation_growth_rate, activation=activation, norm=norm)
            self.decoder_lower = Decoder(lower_dim, int(code_dim / 2), down_t, stride_t, width, depth,
                                         dilation_growth_rate, activation=activation, norm=norm)
        else:
            self.decoder = Decoder(output_dim, code_dim + int(code_dim / 32), down_t, stride_t, width, depth,
                                   dilation_growth_rate, activation=activation, norm=norm)

        self.encoder_upper = Encoder(upper_dim, int(code_dim * 3 / 8), down_t, stride_t, width, depth,
                                     dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_lower = Encoder(lower_dim, int(code_dim * 3 / 8), down_t, stride_t, width, depth,
                                     dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_root = Encoder(root_dim, int(code_dim / 4), down_t, stride_t, width, depth,
                                    dilation_growth_rate, activation=activation, norm=norm)

        self.quantizer_upper = QuantizeEMAReset(nb_code, int(code_dim * 3 / 8), args)
        self.quantizer_lower = QuantizeEMAReset(nb_code, int(code_dim * 3 / 8), args)
        self.quantizer_root = QuantizeEMAReset(nb_code, int(code_dim / 4), args)

        p1_skeletons, p2_skeletons = load_chi3d_group_skeletons()
        self.fk_layers = FK_Pair_Layer_Batch(
            SMPLX_TREE,
            p1_skeletons,
            p2_skeletons,
            args.device
        )

    def mask_group(self, group_cond, force_mask=False):
        if force_mask:
            return torch.zeros_like(group_cond)
        if self.training and self.group_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(group_cond.shape[0], device=group_cond.device) * self.group_mask_prob
            ).view(-1, 1).bool()
            return torch.where(mask, torch.zeros_like(group_cond), group_cond)
        return group_cond

    def _extract_group_pair(self, group_ids):
        if group_ids.dim() == 1:
            group_ids = group_ids.unsqueeze(0)
        if group_ids.shape[-1] == 4:
            p1_group = group_ids[:, 1]
            p2_group = group_ids[:, 3]
        else:
            p1_group = group_ids
            p2_group = group_ids
        if p1_group.min() >= self.group_offset:
            p1_group = p1_group - self.group_offset
            p2_group = p2_group - self.group_offset
        return p1_group.long(), p2_group.long()

    def _get_condition(self, group_ids, force_mask=False):
        p1_group, p2_group = self._extract_group_pair(group_ids)
        p1_emb = self.group_embedding(p1_group)
        p2_emb = self.group_embedding(p2_group)
        cond = torch.cat([p1_emb, p2_emb], dim=1)
        cond = self.mask_group(cond, force_mask=force_mask)
        return self.condition_mlp(cond)

    def forward(self, x, group_ids, type='full'):
        if type == 'full':
            x = x.float()
            upper_emb = self.preprocess(x[..., self.upper_mask])
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb, loss_upper, perplexity_u = self.quantizer_upper(upper_emb)

            lower_emb = self.preprocess(x[..., self.lower_mask])
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb, loss_lower, perplexity_l = self.quantizer_lower(lower_emb)

            root_emb = self.preprocess(x[..., self.root_mask])
            root_emb = self.encoder_root(root_emb)
            root_emb, loss_root, perplexity_r = self.quantizer_root(root_emb)

            loss = loss_upper + loss_lower + loss_root
            perplexity = torch.cat((perplexity_u.unsqueeze(0), perplexity_r.unsqueeze(0), perplexity_l.unsqueeze(0)))

            x_quantized = torch.cat([upper_emb, lower_emb, root_emb], dim=1)
            cond = self._get_condition(group_ids)
            cond = cond.unsqueeze(-1).expand(-1, -1, x_quantized.shape[-1])
            x_quantized = torch.cat([x_quantized, cond], dim=1)

            x_decoder = self.decoder(x_quantized)
            x_out = self.postprocess(x_decoder)
            x_out = self.merge_body_hand(x_out[..., :22 * 12], x_out[..., 22 * 12:52 * 12], x_out[..., 52 * 12:])
            return x_out, loss, perplexity

        if type == 'encode':
            n, _, _ = x.shape
            upper_emb = self.preprocess(x[..., self.upper_mask])
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb = self.postprocess(upper_emb).reshape(-1, upper_emb.shape[1])
            upper_code_idx = self.quantizer_upper.quantize(upper_emb).view(n, -1)

            lower_emb = self.preprocess(x[..., self.lower_mask])
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb = self.postprocess(lower_emb).reshape(-1, lower_emb.shape[1])
            lower_code_idx = self.quantizer_lower.quantize(lower_emb).view(n, -1)

            root_emb = self.preprocess(x[..., self.root_mask])
            root_emb = self.encoder_root(root_emb)
            root_emb = self.postprocess(root_emb).reshape(-1, root_emb.shape[1])
            root_code_idx = self.quantizer_root.quantize(root_emb).view(n, -1)

            return torch.stack([upper_code_idx, lower_code_idx, root_code_idx], dim=-1)

        if type == 'decode':
            x_d_upper = self.quantizer_upper.dequantize(x[..., 0])
            x_d_lower = self.quantizer_lower.dequantize(x[..., 1])
            x_d_root = self.quantizer_root.dequantize(x[..., 2])
            x_d = torch.cat([x_d_upper, x_d_lower, x_d_root], dim=-1).permute(0, 2, 1).contiguous()
            cond = self._get_condition(group_ids)
            cond = cond.unsqueeze(-1).expand(-1, -1, x_d.shape[-1])
            x_d = torch.cat([x_d, cond], dim=1)
            x_decoder = self.decoder(x_d)
            x_out = self.postprocess(x_decoder)
            return self.merge_body_hand(x_out[..., :22 * 12], x_out[..., 22 * 12:52 * 12], x_out[..., 52 * 12:])

        raise ValueError(f"Unsupported type: {type}")

    def merge_body_hand(self, body, hand, root):
        motion = torch.zeros(*body.shape[:2], 56 * 12, device=body.device, dtype=body.dtype)
        motion[..., self.upper_mask] = body
        motion[..., self.lower_mask] = hand
        motion[..., self.root_mask] = root
        return motion

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)



def merge_upper_lower(upper_emb, lower_emb):
    motion = torch.empty(*upper_emb.shape[:2], 263).to(upper_emb.device)
    motion[..., HML_UPPER_BODY_MASK] = upper_emb
    motion[..., HML_LOWER_BODY_MASK] = lower_emb
    return motion
