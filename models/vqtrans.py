import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.motion_transformer import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask, lengths_to_mask
import torch
import torch.distributions as D
from torch import Tensor
from typing import List, Optional, Union
from utils.motion_masks import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK, UPPER_JOINT_Y_MASK, HHI_BODY_ROOT_MASK, HHI_HAND_MASK, HHI_BODY_MASK, HHI_ROOT_MASK_OLD
from utils.fk import ForwardKinematicsLayer, FK_Pair_Layer, ForwardKinematicsLayerBatch, FK_Pair_Layer_Batch
from utils.paramUtil import SMPLX_TREE, INTERX_SKELETONS
from models.trans_encdec import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionEmbeddingLearned1D,
    PositionEmbeddingSine1D,
)

class VQVAE_SEP_shape_primitive(nn.Module):
    def __init__(self,
                 args,
                 nfeats: int,
                 nb_code=512,
                 code_dim = 256,
                 h_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 sep_decoder=False):
        super().__init__()
        if args.dataname == 'kit':
            self.nb_joints = 21
            output_dim = 251
            upper_dim = 120
            lower_dim = 131
        elif args.dataname == 'hhi':
            self.nb_joints = 55
            output_dim = 52 * 12 + 6
            upper_dim = 22 * 12 + 6
            lower_dim = 30 * 12
            self.upper_mask = HHI_BODY_ROOT_MASK
            self.lower_mask = HHI_HAND_MASK
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
        self.query_pos_encoder_upper = PositionEmbeddingLearned1D(self.h_dim)
        self.query_pos_encoder_lower = PositionEmbeddingLearned1D(self.h_dim)
        encoder_layer_upper = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm_upper = nn.LayerNorm(self.h_dim)
        self.encoder_upper = SkipTransformerEncoder(encoder_layer_upper, num_layers,
                                              encoder_norm_upper)
        self.encoder_latent_proj_upper = nn.Linear(self.h_dim, int(code_dim / 2))
        encoder_layer_lower = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm_lower = nn.LayerNorm(self.h_dim)
        self.encoder_lower = SkipTransformerEncoder(encoder_layer_lower, num_layers,
                                              encoder_norm_lower)
        self.encoder_latent_proj_lower = nn.Linear(self.h_dim, int(code_dim / 2))

        self.sep_decoder = sep_decoder
        if self.sep_decoder:
            pass
            # self.decoder_upper = Decoder(upper_dim, int(code_dim / 2), down_t, stride_t, width, depth,
            #                              dilation_growth_rate, activation=activation, norm=norm)
            # self.decoder_lower = Decoder(lower_dim, int(code_dim / 2), down_t, stride_t, width, depth,
            #                              dilation_growth_rate, activation=activation, norm=norm)
        else:

            decoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(decoder_layer, num_layers,
                                              decoder_norm)
            self.decoder_latent_proj = nn.Linear(self.h_dim, output_dim)

        # self.num_code = nb_code
        #
        #
        #
        # self.quantizer_upper = QuantizeEMAReset(nb_code, int(code_dim / 2), args)
        # self.quantizer_lower = QuantizeEMAReset(nb_code, int(code_dim / 2), args)

        interx_skeleton = torch.from_numpy(INTERX_SKELETONS)
        self.fk_layers = FK_Pair_Layer_Batch(
            SMPLX_TREE,
            interx_skeleton[:, 0],
            interx_skeleton[:, 1],
        )


    def forward(self, x, beta, gender, history_motion, type='full'):
        '''type=[full, encode, decode]'''
        if type == 'full':
            x = x.float()
            # x = self.shift_upper_down(x)

            upper_emb = x[..., self.upper_mask]
            lower_emb = x[..., self.lower_mask]
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb, loss_upper, perplexity = self.quantizer_upper(upper_emb)

            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb, loss_lower, perplexity = self.quantizer_lower(lower_emb)
            loss = loss_upper + loss_lower


            ## decoder
            if self.sep_decoder:
                pass

            else:
                x_quantized = torch.cat([upper_emb, lower_emb], dim=1)
                history_motion = history_motion[..., self.upper_mask+self.lower_mask]
                # print(x_quantized.shape)

                gender_embeded = self.gender_embedding(gender).reshape(-1, 8)
                shape_embeded = self.shape_projection(beta)
                history_embeded = self.history_embedding(history_motion)
                combined = torch.cat([gender_embeded, shape_embeded], dim=1)
                condition_vector = self.condition_mlp(combined)
                condition_vector = condition_vector.unsqueeze(-1).expand(-1, -1,
                                                                         x_quantized.shape[-1])  # ([256, 32, 16])
                history_vector = history_embeded.unsqueeze(-1).expand(-1, -1,
                                                                         x_quantized.shape[-1])
                x_quantized = torch.cat([x_quantized, condition_vector, history_vector], dim=1)

                x_decoder = self.decoder(x_quantized)
                x_out = self.postprocess(x_decoder)
                x_out = self.merge_body_hand(x_out[..., :22 * 12 + 6], x_out[..., 22 * 12 + 6:])

            return x_out, loss, perplexity
        elif type == 'encode':
            N, T, _ = x.shape
            x = self.shift_upper_down(x)

            upper_emb = x[..., self.upper_mask]
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb = self.postprocess(upper_emb)
            upper_emb = upper_emb.reshape(-1, upper_emb.shape[-1])
            upper_code_idx = self.quantizer_upper.quantize(upper_emb)
            upper_code_idx = upper_code_idx.view(N, -1)

            lower_emb = x[..., self.lower_mask]
            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb = self.postprocess(lower_emb)
            lower_emb = lower_emb.reshape(-1, lower_emb.shape[-1])
            lower_code_idx = self.quantizer_lower.quantize(lower_emb)
            lower_code_idx = lower_code_idx.view(N, -1)

            code_idx = torch.cat([upper_code_idx.unsqueeze(-1), lower_code_idx.unsqueeze(-1)], dim=-1)
            return code_idx

        elif type == 'decode':
            if self.sep_decoder:
                pass
            else:
                x_d_upper = self.quantizer_upper.dequantize(x[..., 0])
                x_d_lower = self.quantizer_lower.dequantize(x[..., 1])
                x_d = torch.cat([x_d_upper, x_d_lower], dim=-1)
                x_d = x_d.permute(0, 2, 1).contiguous()

                gender_embeded = self.gender_embedding(gender).reshape(-1, 8)
                shape_embeded = self.shape_projection(beta)
                combined = torch.cat([gender_embeded, shape_embeded], dim=1)
                condition_vector = self.condition_mlp(combined)

                condition_vector = condition_vector.tile((1, 1, x_d.shape[-1]))  # ([256, 32, 16])
                x_d = torch.cat([x_d, condition_vector], dim=1)

                x_decoder = self.decoder(x_d)
                x_out = self.postprocess(x_decoder)
                x_out = self.merge_body_hand(x_out[..., :22 * 12 + 6], x_out[..., 22 * 12 + 6:])

                return x_out

    def merge_body_hand(self, body, hand):
        motion = torch.zeros(*body.shape[:2], 56 * 12).to(body.device)
        motion[..., HHI_BODY_ROOT_MASK] = body
        motion[..., HHI_HAND_MASK] = hand
        return motion

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x



class DualPersonMldVAE(nn.Module):
    def __init__(self,
                 args,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # === 1. 定义三部分的维度 (双人 = x2) ===
        # 我们假设输入时，P1和P2的对应部分已经拼在一起，或者在forward里拼
        if args.dataname == 'hhi':
            self.dim_trans = 3 * 2
            self.dim_body = 22 * 6 * 2
            self.dim_hand = 30 * 6 * 2
            self.body_mask = HHI_BODY_MASK
            self.hand_mask = HHI_HAND_MASK
            self.root_mask = HHI_ROOT_MASK_OLD
        else:
            self.nb_joints = 22
            output_dim = 263
            upper_dim = 156
            lower_dim = 107
            self.upper_mask = HML_UPPER_BODY_MASK
            self.lower_mask = HML_LOWER_BODY_MASK


        self.h_dim = latent_dim  # Transformer 的隐藏层维度

        # === 2. 独立嵌入层 (Respect Different Modes) ===
        # 位移、身体、手部的数据分布差异大，先各自过一层 Linear 映射到特征空间
        # 我们分配 hidden dim 的一部分给它们，或者映射后拼接
        # 这里策略是：分别映射到 h_dim，然后加权或拼接。
        # 为了让 Transformer 处理，通常是将它们映射后相加，或者拼接后由一层 Linear 融合。
        # 这里采用：Concat(Emb(T), Emb(B), Emb(H)) -> Linear -> Transformer Input
        self.query_pos_encoder = PositionEmbeddingLearned1D(
            self.latent_dim)
        self.query_pos_decoder = PositionEmbeddingLearned1D(
            self.latent_dim)

        self.emb_trans = nn.Linear(self.dim_trans, 64)  # 位移特征通常较少，维度给小点
        self.emb_body = nn.Linear(self.dim_body, 512)  # 身体是主体
        self.emb_hand = nn.Linear(self.dim_hand, 256)  # 手部细节

        # 融合层：将分开 Embedding 的特征融合为 Transformer 的输入维度
        self.fusion_layer = nn.Linear(64 + 512 + 256, self.h_dim)

        # === 3. 共享 Encoder (用于学习双人交互和全身协调) ===
        # MLD 风格：Global Token
        self.global_motion_token = nn.Parameter(torch.randn(1, self.h_dim))

        encoder_layer = TransformerEncoderLayer(
            self.h_dim, num_heads, ff_size, dropout, activation
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)

        # === 4. 三个独立的 Latent Heads (分离潜空间) ===
        # 输出均值和方差 (2 * latent_dim)
        # 我们可以让 z_trans, z_body, z_hand 的维度不同，也可以相同。这里设为相同以便处理。
        self.proj_z_trans = nn.Linear(self.h_dim, 2 * int(self.latent_dim/4))
        self.proj_z_body = nn.Linear(self.h_dim, 2 * self.latent_dim)
        self.proj_z_hand = nn.Linear(self.h_dim, 2 * self.latent_dim)

        # === 5. Decoder (共享 Backbone，独立输出 Head) ===
        # 将三个 z 拼回去
        gender_emb_dim = int(latent_dim/8)
        shape_emb_dim = int(latent_dim/2)
        self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gender_emb_dim)  # 2个类别: 0和1
        self.shape_projection = nn.Linear(20, shape_emb_dim)  # 10是你的体型向量维度
        combined_input_dim = 2 * gender_emb_dim + shape_emb_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.z_fusion = nn.Linear(2 * self.latent_dim + int(self.latent_dim/4), self.h_dim)

        # 也可以选择用三个独立的 Decoder，但那样交互性差。建议用共享 Decoder。
        decoder_layer = TransformerEncoderLayer(
            self.h_dim, num_heads, ff_size, dropout, activation
        )
        decoder_norm = nn.LayerNorm(self.h_dim)
        self.decoder = SkipTransformerEncoder(decoder_layer, num_layers, decoder_norm)

        # === 6. 独立的输出层 ===
        self.final_trans = nn.Linear(self.h_dim, self.dim_trans)
        self.final_body = nn.Linear(self.h_dim, self.dim_body)
        self.final_hand = nn.Linear(self.h_dim, self.dim_hand)

        # fk_loss
        interx_skeleton = torch.from_numpy(INTERX_SKELETONS)
        self.fk_layers = FK_Pair_Layer_Batch(
            SMPLX_TREE,
            interx_skeleton[:, 0],
            interx_skeleton[:, 1],
        )

    def forward(self, x, beta, gender, lengths: Optional[List[int]] = None):
        """
        Input:
            x_trans: (N, T, dim_trans_total)  <- P1+P2
            x_body:  (N, T, dim_body_total)   <- P1+P2
            x_hand:  (N, T, dim_hand_total)   <- P1+P2
        """
        # 1. Encode -> 得到三个分开的 z 和分布
        # z_parts = (z_trans, z_body, z_hand)
        if lengths is None:
            lengths = [x.shape[1]] * x.shape[0]


        z_parts, dists = self.encode(x, lengths)

        # 2. Decode -> 传入 z 组合，还原三部分数据
        # 注意：连续 VAE 需要长度 T
        recon_motion = self.decode(z_parts, beta, gender, lengths=lengths)

        return recon_motion, z_parts, dists

    def encode(self, x, lengths: Optional[List[int]] = None):
        N, T, _ = x.shape
        device = x.device
        x_body = x[..., self.body_mask]
        x_hand = x[..., self.hand_mask]
        x_trans = x[..., self.root_mask]

        mask = lengths_to_mask(lengths, device)
        token_mask = torch.ones((N, 1), dtype=torch.bool, device=device)
        aug_mask = torch.cat((token_mask, mask), dim=1)  # (N, T+1)

        # --- A. 独立处理不同模式的特征 (Indep. Embedding) ---
        # 保证位移的线性特性不被旋转特征干扰
        feat_trans = self.emb_trans(x_trans.float())  # (N, T, 64)
        feat_body = self.emb_body(x_body.float())  # (N, T, 256)
        feat_hand = self.emb_hand(x_hand.float())  # (N, T, 192)

        # --- B. 融合 (Concat & Fuse) ---
        feat_all = torch.cat([feat_trans, feat_body, feat_hand], dim=-1)  # (N, T, 512)
        feat_fused = self.fusion_layer(feat_all)  # (N, T, h_dim)

        # --- C. 转为 Transformer 格式 (Time First) ---
        feat_seq = feat_fused.permute(1, 0, 2).contiguous()  # (T, N, h_dim)


        # --- D. 插入 Global Token ---
        token = self.global_motion_token.unsqueeze(1).repeat(1, N, 1)  # (1, N, h_dim)
        x_seq = torch.cat((token, feat_seq), dim=0)  # (T+1, N, h_dim)
        x_seq = self.query_pos_encoder(x_seq)

        # --- E. 交互编码 (Encoder) ---
        # 此时 Transformer 内部会计算 P1/P2、身体/手/位移 之间的所有 Attention
        out_seq = self.encoder(x_seq, src_key_padding_mask=~aug_mask)

        # --- F. 提取 Global Feature ---
        global_feat = out_seq[:1]  # (N, h_dim)

        # --- G. 分离潜空间 (Split Latent Space) ---
        # 1. Translation Distribution
        params_trans = self.proj_z_trans(global_feat)
        z_t, dist_t = self.reparameterize(params_trans)

        # 2. Body Distribution
        params_body = self.proj_z_body(global_feat)
        z_b, dist_b = self.reparameterize(params_body)

        # 3. Hand Distribution
        params_hand = self.proj_z_hand(global_feat)
        z_h, dist_h = self.reparameterize(params_hand)
        # print("z_t shape:", z_t.shape, "z_b shape:", z_b.shape, "z_h shape:", z_h.shape)

        return (z_t, z_b, z_h), [dist_t, dist_b, dist_h]

    def decode(self, z_parts, beta, gender, lengths):
        z_t, z_b, z_h = z_parts
        device = z_t.device
        mask = lengths_to_mask(lengths, device)
        N, T = mask.shape
        cond_mask = torch.ones((N, 1), dtype=torch.bool, device=device)
        aug_mask = torch.cat((cond_mask, cond_mask, mask), dim=1)  # (N, T+1)

        # --- A. 融合潜变量 (Latent Fusion) ---
        # 虽然 latent 是分开的，但解码时需要知道彼此的状态
        # 比如：手的位置取决于身体，身体的位置取决于位移
        z_concat = torch.cat([z_t, z_b, z_h], dim=-1)  # (N, 3*latent_dim)
        # print('z_concat shape:', z_concat.shape)
        z_fused = self.z_fusion(z_concat)  # (N, h_dim)

        # --- B. 时间维度扩展 (Expand over Time) ---
        # MLD 方式：将 Z 重复 T 次
        query = torch.zeros(T, N, self.latent_dim, device=z_fused.device)
        # x_d = z_fused.unsqueeze(0).repeat(T, 1, 1)  # (T, N, h_dim)

        # (可选) Add Positional Encoding here
        gender_embeded = self.gender_embedding(gender).reshape(N, -1)
        shape_embeded = self.shape_projection(beta)
        combined = torch.cat([gender_embeded, shape_embeded], dim=1)
        condition_vector = self.condition_mlp(combined).unsqueeze(0)
        x_d = torch.cat([z_fused, condition_vector, query], dim=0)
        x_d = self.query_pos_decoder(x_d)

        # --- C. 解码 (Shared Decoder) ---
        # 恢复动作序列的时序细节
        dec_out = self.decoder(x_d, src_key_padding_mask=~aug_mask)  # (T, N, h_dim)
        dec_out = dec_out[2:]
        # 转回 Batch First
        dec_out = dec_out.permute(1, 0, 2).contiguous()  # (N, T, h_dim)

        # --- D. 独立输出头 (Independent Output Heads) ---
        # 每一部分由专门的 Linear 层回归，互不干扰数据分布
        recon_trans = self.final_trans(dec_out)
        recon_body = self.final_body(dec_out)
        recon_hand = self.final_hand(dec_out)

        out = self.merge_body_hand(recon_body, recon_hand, recon_trans)

        return out

    def reparameterize(self, params):
        mu, logvar = torch.chunk(params, 2, dim=-1)
        std = logvar.exp().pow(0.5)
        dist = D.Normal(mu, std)
        z = dist.rsample()
        return z, dist

    def merge_body_hand(self, body, hand, root):

        motion = torch.zeros(*body.shape[:2], 56 * 12).to(body.device)
        motion[..., HHI_BODY_MASK] = body
        motion[..., HHI_HAND_MASK] = hand
        motion[..., HHI_ROOT_MASK_OLD] = root
        return motion


class MldVae(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 **kwargs) -> None:
        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]

        input_feats = nfeats
        output_feats = nfeats


        self.query_pos_encoder = PositionEmbeddingLearned1D(self.latent_dim)
        self.query_pos_decoder = PositionEmbeddingLearned1D(self.latent_dim)

        gender_emb_dim = int(self.latent_dim/8)
        shape_emb_dim = int(self.latent_dim/2)
        self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gender_emb_dim)  # 2个类别: 0和1
        self.shape_projection = nn.Linear(20, shape_emb_dim)  # 10是你的体型向量维度
        combined_input_dim = 2 * gender_emb_dim + shape_emb_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                              decoder_norm)

        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

        interx_skeleton = torch.from_numpy(INTERX_SKELETONS)
        self.fk_layers = FK_Pair_Layer_Batch(
            SMPLX_TREE,
            interx_skeleton[:, 0],
            interx_skeleton[:, 1],
        )

    def forward(self, features: Tensor, beta, gender, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        # print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, beta, gender, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, D]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq,
                            src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist

    def decode(self, z: Tensor, beta, gender, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        queries = self.query_pos_decoder(queries)
        # mem_pos = self.mem_pos_decoder(z)
        gender_embeded = self.gender_embedding(gender).reshape(bs, -1)
        shape_embeded = self.shape_projection(beta)
        combined = torch.cat([gender_embeded, shape_embeded], dim=1)
        condition_vector = self.condition_mlp(combined).unsqueeze(0)
        z_conditioned = torch.cat([z, condition_vector], dim=0)

        output = self.decoder(
            tgt=queries,
            memory=z_conditioned,
            tgt_key_padding_mask=~mask,
            # query_pos=query_pos,
            # pos=mem_pos,
        ).squeeze(0)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats

class AutoMldVae(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: tuple = [1, 256],
                 h_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.h_dim = h_dim
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False
        self.pe_type = "mld"

        self.query_pos_encoder = PositionEmbeddingLearned1D(
            self.h_dim)
        self.query_pos_decoder = PositionEmbeddingLearned1D(
            self.h_dim)

        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        gender_emb_dim = int(self.latent_dim/8)
        shape_emb_dim = int(self.latent_dim/2)
        self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gender_emb_dim)  # 2个类别: 0和1
        self.shape_projection = nn.Linear(20, shape_emb_dim)  # 10是你的体型向量维度
        combined_input_dim = 2 * gender_emb_dim + shape_emb_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.h_dim)
        )

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")
        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)

        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.h_dim))

        self.skel_embedding = nn.Linear(input_feats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, output_feats)

        self.register_buffer('latent_mean', torch.tensor(0))
        self.register_buffer('latent_std', torch.tensor(1))
        interx_skeleton = torch.from_numpy(INTERX_SKELETONS)
        self.fk_layers = FK_Pair_Layer_Batch(
            SMPLX_TREE,
            interx_skeleton[:, 0],
            interx_skeleton[:, 1],
        )

    def encode(
            self,
            future_motion, history_motion,
            scale_latent: bool = False,
    ) -> Union[Tensor, D]:
        device = future_motion.device
        bs, nfuture, nfeats = future_motion.shape
        nhistory = history_motion.shape[1]

        x = torch.cat((history_motion, future_motion), dim=1)  # [bs, H+F, nfeats]
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, h_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq)[:dist.shape[0]]  # [2*latent_size, bs, h_dim]
        dist = self.encoder_latent_proj(dist)  # [2*latent_size, bs, latent_dim]

        # content distribution
        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]
        logvar = torch.clamp(logvar, min=-10, max=10)  # avoid numerical issues, otherwise denoiser rollout can break
        # if torch.isnan(mu).any() or torch.isinf(mu).any() or torch.isnan(logvar).any() or torch.isinf(logvar).any():
        #     pdb.set_trace()

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        if scale_latent:  # only used during denoiser training
            latent = latent / self.latent_std
        return latent, dist

    def decode(self, z: Tensor, history_motion, nfuture, beta, gender,
               scale_latent: bool = False,
               ):
        bs = history_motion.shape[0]
        if scale_latent:  # only used during denoiser training
            z = z * self.latent_std
        z = self.decoder_latent_proj(z)  # [latent_size, bs, latent_dim] => [latent_size, bs, h_dim]
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2)  # [nhistory, bs, h_dim]

        # Pass through the transformer decoder
        # with the latent vector for memory
        gender_embeded = self.gender_embedding(gender).reshape(bs, -1)
        shape_embeded = self.shape_projection(beta)
        combined = torch.cat([gender_embeded, shape_embeded], dim=1)
        condition_vector = self.condition_mlp(combined).unsqueeze(0)

        if self.arch == "all_encoder":
            # print('condition_vector shape:', condition_vector.shape, 'z shape:', z.shape, 'history_embedding shape:', history_embedding.shape, 'queries shape:', queries.shape)
            xseq = torch.cat((condition_vector, z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                xseq)[-nfuture:]

        elif self.arch == "encoder_decoder":
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            z_conditioned = torch.cat([z, condition_vector], dim=0)
            output = self.decoder(
                tgt=xseq,
                memory=z_conditioned,
            )
            # print('output:', output.shape)
            output = output[-nfuture:]

        output = self.final_layer(output)
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats


class MldVaeSeparate(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 **kwargs) -> None:
        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.trans_h_dim = int(self.latent_dim/4)

        self.dim_trans = 3 * 2
        self.dim_body = 22 * 6 * 2
        self.dim_hand = 30 * 6 * 2
        output_feats = self.dim_trans + self.dim_body + self.dim_hand
        self.body_mask = HHI_BODY_MASK
        self.hand_mask = HHI_HAND_MASK
        self.root_mask = HHI_ROOT_MASK_OLD

        self.query_pos_encoder_body = PositionEmbeddingLearned1D(self.latent_dim)
        self.query_pos_encoder_hand = PositionEmbeddingLearned1D(self.latent_dim)
        self.query_pos_encoder_trans = PositionEmbeddingLearned1D(self.trans_h_dim)
        self.query_pos_decoder = PositionEmbeddingLearned1D(self.latent_dim)

        gender_emb_dim = int(self.latent_dim/8)
        shape_emb_dim = int(self.latent_dim/2)
        self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gender_emb_dim)  # 2个类别: 0和1
        self.shape_projection = nn.Linear(20, shape_emb_dim)  # 10是你的体型向量维度
        combined_input_dim = 2 * gender_emb_dim + shape_emb_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_layer_trans = TransformerEncoderLayer(
            self.trans_h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        encoder_norm_trans = nn.LayerNorm(self.trans_h_dim)
        self.encoder_body = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        self.encoder_hand = SkipTransformerEncoder(encoder_layer, num_layers,
                                                   encoder_norm)
        self.encoder_trans = SkipTransformerEncoder(encoder_layer_trans, num_layers,
                                                   encoder_norm_trans)


        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                              decoder_norm)

        self.global_body_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.latent_dim))
        self.global_hand_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.latent_dim))
        self.global_trans_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.trans_h_dim))

        self.body_embedding = nn.Linear(self.dim_body, self.latent_dim)
        self.hand_embedding = nn.Linear(self.dim_hand, self.latent_dim)
        self.trans_embedding = nn.Linear(self.dim_trans, self.trans_h_dim)
        self.latent_fusion = nn.Linear(self.latent_dim * 2 + self.trans_h_dim, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

        interx_skeleton = torch.from_numpy(INTERX_SKELETONS)
        self.fk_layers = FK_Pair_Layer_Batch(
            SMPLX_TREE,
            interx_skeleton[:, 0],
            interx_skeleton[:, 1],
        )

    def forward(self, features: Tensor, beta, gender, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        # print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, beta, gender, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, D]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        x_body = x[..., self.body_mask]
        x_hand = x[..., self.hand_mask]
        x_trans = x[..., self.root_mask]
        # Embed each human poses into latent vectors
        x_body = self.body_embedding(x_body)
        x_hand = self.hand_embedding(x_hand)
        x_trans = self.trans_embedding(x_trans)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x_body = x_body.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]
        x_hand = x_hand.permute(1, 0, 2)
        x_trans = x_trans.permute(1, 0, 2)

        # Each batch has its own set of tokens
        dist_body = torch.tile(self.global_body_token[:, None, :], (1, bs, 1))
        dist_hand = torch.tile(self.global_hand_token[:, None, :], (1, bs, 1))
        dist_trans = torch.tile(self.global_trans_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist_body.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq_body = torch.cat((dist_body, x_body), 0)
        xseq_hand = torch.cat((dist_hand, x_hand), 0)
        xseq_trans = torch.cat((dist_trans, x_trans), 0)

        xseq_body = self.query_pos_encoder_body(xseq_body)
        xseq_hand = self.query_pos_encoder_hand(xseq_hand)
        xseq_trans = self.query_pos_encoder_trans(xseq_trans)

        dist_body = self.encoder_body(xseq_body,
                            src_key_padding_mask=~aug_mask)[:dist_body.shape[0]]
        dist_hand = self.encoder_hand(xseq_hand,
                            src_key_padding_mask=~aug_mask)[:dist_hand.shape[0]]
        dist_trans = self.encoder_trans(xseq_trans,
                            src_key_padding_mask=~aug_mask)[:dist_trans.shape[0]]


        mu_body = dist_body[0:self.latent_size, ...]
        logvar_body = dist_body[self.latent_size:, ...]
        mu_hand = dist_hand[0:self.latent_size, ...]
        logvar_hand = dist_hand[self.latent_size:, ...]
        mu_trans = dist_trans[0:self.latent_size, ...]
        logvar_trans = dist_trans[self.latent_size:, ...]

        # resampling
        std_body = logvar_body.exp().pow(0.5)
        dist_body = torch.distributions.Normal(mu_body, std_body)
        latent_body = dist_body.rsample()

        std_hand = logvar_hand.exp().pow(0.5)
        dist_hand = torch.distributions.Normal(mu_hand, std_hand)
        latent_hand = dist_hand.rsample()

        std_trans = logvar_trans.exp().pow(0.5)
        dist_trans = torch.distributions.Normal(mu_trans, std_trans)
        latent_trans = dist_trans.rsample()
        return (latent_body, latent_hand, latent_trans), [dist_body, dist_hand, dist_trans]

    def decode(self, z, beta, gender, lengths: List[int]):

        z_body, z_hand, z_trans = z
        z = torch.cat([z_body, z_hand, z_trans], dim=-1)
        z = self.latent_fusion(z)

        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        queries = self.query_pos_decoder(queries)
        # mem_pos = self.mem_pos_decoder(z)
        gender_embeded = self.gender_embedding(gender).reshape(bs, -1)
        shape_embeded = self.shape_projection(beta)
        combined = torch.cat([gender_embeded, shape_embeded], dim=1)
        condition_vector = self.condition_mlp(combined).unsqueeze(0)
        z_conditioned = torch.cat([z, condition_vector], dim=0)

        output = self.decoder(
            tgt=queries,
            memory=z_conditioned,
            tgt_key_padding_mask=~mask,
            # query_pos=query_pos,
            # pos=mem_pos,
        ).squeeze(0)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        feats = self.merge_body_hand(feats[..., :22 * 12], feats[..., 22 * 12:52 * 12], feats[..., 52 * 12:])
        return feats

    def merge_body_hand(self, body, hand, root):

        motion = torch.zeros(*body.shape[:2], 56 * 12).to(body.device)
        motion[..., HHI_BODY_MASK] = body
        motion[..., HHI_HAND_MASK] = hand
        motion[..., HHI_ROOT_MASK_OLD] = root
        return motion
