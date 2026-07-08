import torch.nn as nn
import torch.nn.functional as F
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask
import torch
from utils.humanml_utils import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK, UPPER_JOINT_Y_MASK, HHI_BODY_ROOT_MASK, HHI_HAND_ROOT_MASK, HHI_ROOT_MASK, HHI_HAND_MASK, HHI_BODY_MASK
from utils.fk import ForwardKinematicsLayer, FK_Pair_Layer, ForwardKinematicsLayerBatch, FK_Pair_Layer_Batch
from utils.paramUtil import SMPLX_TREE, INTERX_SKELETONS


class Hierarchical_VQVAE_TwoPerson(nn.Module):

    def __init__(self,
                 input_feats: int,
                 output_feats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=1024,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:
        super().__init__()
        self.single_person_feats = input_feats // 2
        self.code_dim_t = code_dim * 2
        self.code_dim_b = code_dim
        self.code_num = code_num
        self.output_emb_width = output_emb_width
        self.latent_dim = (output_emb_width // 2) + 200
        self.down_t = down_t
        self.stride_t = stride_t
        self.width = width
        self.depth = depth
        self.cond_drop_prob = 0.1

        num_genre = 10
        mfcc_dim = 40
        chroma_dim = 12
        self.twoperson_encoder = nn.Conv1d(2, 1, 3, 1, 1)
        self.encoder_b = Encoder(self.single_person_feats,
                                 output_emb_width,
                                 down_t,
                                 stride_t,
                                 width,
                                 depth,
                                 dilation_growth_rate,
                                 activation=activation,
                                 norm=norm)
        self.encoder_t = Encoder(output_emb_width,
                                 self.code_dim_t,
                                 down_t - 1,
                                 stride_t,
                                 width,
                                 depth,
                                 dilation_growth_rate,
                                 activation=activation,
                                 norm=norm)

        self.quantize_conv_b = nn.Conv1d(2 * self.output_emb_width, self.code_dim_b, 3, 1, 1)
        self.upsample_t = nn.ConvTranspose1d(self.code_dim_t, self.code_dim_b, kernel_size=2, stride=2)
        self.mfcc_encoder_b = Encoder(mfcc_dim,
                                      self.output_emb_width // 8,
                                      down_t + 1,
                                      stride_t,
                                      width,
                                      depth,
                                      dilation_growth_rate,
                                      activation=activation,
                                      norm=norm)
        self.chroma_encoder_b = Encoder(chroma_dim,
                                        self.output_emb_width // 8,
                                        down_t + 1,
                                        stride_t,
                                        width,
                                        depth,
                                        dilation_growth_rate,
                                        activation=activation,
                                        norm=norm)

        self.genre_emb_dim = kwargs['genre_emb_dim']
        self.genre_emb = nn.Linear(num_genre, self.genre_emb_dim)
        self.music_feats_enc_dim_b = (output_emb_width // 8) + (output_emb_width // 8)

        self.quantizer_t = QuantizeEMAReset(code_num, self.code_dim_t, mu=0.99, gpu_id=kwargs['gpu_id'])
        self.quantizer_b = QuantizeEMAReset(code_num, self.code_dim_b, mu=0.99, gpu_id=kwargs['gpu_id'])

        self.decoder_t = Decoder(self.output_emb_width,
                                 self.code_dim_t,
                                 down_t - 1,
                                 stride_t,
                                 width,
                                 depth,
                                 dilation_growth_rate,
                                 activation=activation,
                                 norm=norm)
        self.decoder_b = Decoder(output_feats,
                                 self.code_dim_b + self.code_dim_b + self.music_feats_enc_dim_b + self.genre_emb_dim,
                                 down_t,
                                 stride_t,
                                 width,
                                 depth,
                                 dilation_growth_rate,
                                 activation=activation,
                                 norm=norm)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, two_person_features, genre, mfcc_feats, chroma_feats):
        B, seq_len, two_person_dim = two_person_features.shape
        twoperson_feats_reshape = torch.cat((two_person_features[..., :self.single_person_feats].unsqueeze(-1),
                                             two_person_features[..., self.single_person_feats:].unsqueeze(-1)), -1)
        twoperson_feats_reshape = twoperson_feats_reshape.view(B * seq_len, 2, -1)
        features = self.twoperson_encoder(twoperson_feats_reshape).view(B, seq_len, -1)
        features = self.preprocess(features)
        # Top level encoding and quantization
        enc_b = self.encoder_b(features)
        encoder_t = self.encoder_t(enc_b)

        top_token_len = encoder_t.shape[-1]
        x_quantized_t, loss_t, perplexity_t, code_idx_t = self.quantizer_t(encoder_t)
        decoder_t = self.decoder_t(x_quantized_t)
        upsample_t = self.upsample_t(x_quantized_t)

        # Bottom level encoding and quantization
        encoder_b = torch.cat([decoder_t, enc_b], dim=1)
        quant_b = self.quantize_conv_b(encoder_b)
        bottom_token_len = quant_b.shape[-1]
        x_quantized_b, loss_b, perplexity_b, code_idx_b = self.quantizer_b(quant_b)

        # Encoding the conditions for bottom level decoding
        randomizer = torch.randint(0, 2, (1, 1))
        if self.training and randomizer == 1:
            genre = torch.zeros_like(genre).to(features.device)
        genre_emb = self.genre_emb(genre.float().to(features.device)).reshape(len(genre), self.genre_emb_dim, 1)
        genre_emb_b = genre_emb.repeat(1, 1, bottom_token_len)
        mfcc_in_b = self.mfcc_encoder_b(self.preprocess(mfcc_feats))
        chroma_in_b = self.chroma_encoder_b(self.preprocess(chroma_feats))

        x_cond_b = torch.cat([upsample_t, x_quantized_b, mfcc_in_b, chroma_in_b, genre_emb_b], dim=1)
        x_out = self.preprocess(self.decoder_b(x_cond_b))
        perplexity = torch.cat((perplexity_b.unsqueeze(0), perplexity_t.unsqueeze(0)))
        return x_out, (loss_b + loss_t), perplexity, [code_idx_b, code_idx_t]

    def encode(self, two_person_features):
        B, seq_len, two_person_dim = two_person_features.shape
        twoperson_feats_reshape = torch.cat((two_person_features[..., :self.single_person_feats].unsqueeze(-1),
                                             two_person_features[..., self.single_person_feats:].unsqueeze(-1)), -1)
        twoperson_feats_reshape = twoperson_feats_reshape.view(B * seq_len, 2, -1)
        features = self.twoperson_encoder(twoperson_feats_reshape).view(B, seq_len, -1)
        features = self.preprocess(features)
        # Top level encoding and quantization
        enc_b = self.encoder_b(features)
        encoder_t = self.encoder_t(enc_b)

        x_quantized_t, loss_t, perplexity_t, code_idx_t = self.quantizer_t(encoder_t)
        decoder_t = self.decoder_t(x_quantized_t)

        # Bottom level encoding and quantization
        encoder_b = torch.cat([decoder_t, enc_b], dim=1)
        quant_b = self.quantize_conv_b(encoder_b)
        bottom_token_len = quant_b.shape[-1]
        x_quantized_b, loss_b, perplexity_b, code_idx_b = self.quantizer_b(quant_b)
        code_idx = [code_idx_b, code_idx_t]
        return code_idx

    def decode(self, code_idx, genre, mfcc_feats, chroma_feats):
        code_idx_b = code_idx[0]
        code_idx_t = code_idx[1]
        B, bottom_token_len = code_idx_b.shape
        x_quantized_t = self.quantizer_t.dequantize(code_idx_t)
        x_quantized_t = x_quantized_t.view(B, -1, self.code_dim_t).permute(0, 2, 1).contiguous()
        x_quantized_b = self.quantizer_b.dequantize(code_idx_b)
        x_quantized_b = x_quantized_b.view(B, -1, self.code_dim_b).permute(0, 2, 1).contiguous()
        upsample_t = self.upsample_t(x_quantized_t)

        genre = torch.zeros_like(genre).to(code_idx_b.device)
        genre_emb = self.genre_emb(genre.float().to(code_idx_b.device)).reshape(len(genre), self.genre_emb_dim, 1)
        genre_emb_b = genre_emb.repeat(1, 1, bottom_token_len)
        mfcc_in_b = self.mfcc_encoder_b(self.preprocess(mfcc_feats))
        chroma_in_b = self.chroma_encoder_b(self.preprocess(chroma_feats))
        x_cond_b = torch.cat([upsample_t, x_quantized_b, mfcc_in_b, chroma_in_b, genre_emb_b], dim=1)
        x_out = self.preprocess(self.decoder_b(x_cond_b))
        return x_out


class VQVAE_672_shape(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 level=None):

        super().__init__()
        self.code_dim_b = code_dim
        self.code_dim_t = code_dim * 2
        self.num_code = nb_code
        self.quant = args.quantizer
        output_dim = 56 * 2 * 6  # 55 joints, 1 trans, 2 person, 6d rotation
        self.encoder_b = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)
        self.encoder_t = Encoder(output_emb_width, self.code_dim_t, down_t-1, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)

        gender_emb_dim = 4
        shape_emb_dim = 40
        self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gender_emb_dim)  # 2个类别: 0和1
        self.shape_projection = nn.Linear(20, shape_emb_dim)  # 10是你的体型向量维度
        # 3. 深度融合分支 (MLP)
        combined_input_dim = 2 * gender_emb_dim + shape_emb_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )

        self.decoder_b = Decoder(output_dim, self.code_dim_b+self.code_dim_b+32, down_t, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)
        self.decoder_t = Decoder(output_emb_width, self.code_dim_t, down_t-1, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)

        self.quantize_conv_b = nn.Conv1d(2 * output_emb_width, self.code_dim_b, 3, 1, 1)
        self.upsample_t = nn.ConvTranspose1d(self.code_dim_t, self.code_dim_b, kernel_size=2, stride=2)

        if args.quantizer == "ema_reset":
            self.quantizer_b = QuantizeEMAReset(nb_code, self.code_dim_b, args)
            self.quantizer_t = QuantizeEMAReset(nb_code, self.code_dim_t, args)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x, beta, gender):

        x_in = self.preprocess(x)
        # print('x_in shape:', x_in.shape)

        x_encoder_b = self.encoder_b(x_in)
        # print('x_encoder_b shape:', x_encoder_b.shape)
        x_encoder_t = self.encoder_t(x_encoder_b)
        # print('x_encoder_t shape:', x_encoder_t.shape)

        ## quantization top
        x_quantized_t, loss_t, perplexity_t = self.quantizer_t(x_encoder_t)

        ## decoder top
        decoder_t = self.decoder_t(x_quantized_t)
        # print('decoder_t shape:', decoder_t.shape)
        upsample_t = self.upsample_t(x_quantized_t)
        # print('upsample_t shape:', upsample_t.shape)

        ## quantization bottom
        # target_T_b = x_encoder_b.shape[-1]  # 获取 x_encoder_b 的长度 (37)
        # # 在最后一个维度（时间）的右侧填充 (0, 1) [左边0, 右边1]
        # decoder_t_padded = F.pad(decoder_t, (0, target_T_b - decoder_t.shape[-1]))

        encoder_b = torch.cat([decoder_t, x_encoder_b], dim=1)

        # encoder_b = torch.cat([decoder_t, x_encoder_b], dim=1)
        # print('encoder_b shape:', encoder_b.shape)
        quant_b = self.quantize_conv_b(encoder_b)
        # print('quant_b shape:', quant_b.shape)
        x_quantized_b, loss_b, perplexity_b = self.quantizer_b(quant_b)

        ## decoder bottom with condition
        gender_embeded = self.gender_embedding(gender).reshape(-1, 8)
        shape_embeded = self.shape_projection(beta)
        combined = torch.cat([gender_embeded, shape_embeded], dim=1)
        condition_vector = self.condition_mlp(combined)
        condition_vector = condition_vector.unsqueeze(-1).expand(-1, -1, x_quantized_b.shape[-1])  # ([256, 32, 16])

        # target_T_b = x_quantized_b.shape[-1]  # 动态获取目标长度
        # padding_needed = target_T_b - upsample_t.shape[-1]
        # upsample_t_padded = F.pad(upsample_t, (0, padding_needed))  # 填充量是计算出来的
        # print('x_quantized_b shape:', x_quantized_b.shape, 'upsample_t shape:', upsample_t.shape, 'condition_vector shape:', condition_vector.shape)
        x_quantized_b_cond = torch.cat([x_quantized_b, upsample_t, condition_vector], dim=1)

        x_decoder = self.decoder_b(x_quantized_b_cond)
        x_out = self.postprocess(x_decoder)
        perplexity = torch.cat((perplexity_b.unsqueeze(0), perplexity_t.unsqueeze(0)))
        return x_out, (loss_b + loss_t), perplexity

    def forward_decoder(self, x, beta, gender):

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        gender_embeded = self.gender_embedding(gender).reshape(-1, 8)
        shape_embeded = self.shape_projection(beta)
        combined = torch.cat([gender_embeded, shape_embeded], dim=1)
        condition_vector = self.condition_mlp(combined)

        condition_vector = condition_vector.tile((1, 1, x_d.shape[-1]))  # ([256, 32, 16])
        x_d = torch.cat([x_d, condition_vector], dim=1)

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanVQVAEHI(nn.Module):
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
                 level=None):

        super().__init__()

        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_672_shape(args, nb_code, code_dim, code_dim, down_t, stride_t, width, depth,
                                     dilation_growth_rate, activation=activation, norm=norm, level=level)
        interx_skeleton = torch.from_numpy(INTERX_SKELETONS)
        self.fk_layers = FK_Pair_Layer_Batch(
            SMPLX_TREE,
            interx_skeleton[:, 0],
            interx_skeleton[:, 1],
        )

    def forward(self, x, betas, gender, type='full'):
        '''type=[full, encode, decode]'''
        if type == 'full':
            x_out, loss, perplexity = self.vqvae(x, betas, gender)
            return x_out, loss, perplexity
        elif type == 'encode':
            b, t, c = x.size()
            quants = self.vqvae.encode(x)  # (N, T)
            return quants
        elif type == 'decode':
            x_out = self.vqvae.forward_decoder(x, betas, gender)
            return x_out
        else:
            raise ValueError(f'Unknown "{type}" type')