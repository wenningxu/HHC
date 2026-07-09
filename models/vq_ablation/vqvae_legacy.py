import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset, FSQ
from models.motion_transformer import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask
from utils.fk import FK_Single_Layer_Batch
from utils.paramUtil import SMPLX_TREE, INTERX_SKELETONS
import torch
import sys

class VQVAE_672(nn.Module):
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
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        output_dim = 56*2*6  # 55 joints, 1 trans, 2 person, 6d rotation
        self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        # Transformer Encoder
        # self.encoder = Encoder_Transformer(
        #     input_feats=output_dim,
        #     embed_dim=512, # 1024
        #     output_dim=512,
        #     block_size=4,
        #     num_layers=6,
        #     n_head=16
        # )

         # Transformer Encoder 4 frames
        # from exit.motiontransformer import MotionTransformerEncoder
        # in_feature = 251 if args.dataname == 'kit' else 263
        # self.encoder2 = MotionTransformerEncoder(in_feature, args.code_dim, num_frames=4, num_layers=2)

        self.decoder = Decoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # self.decoder = Decoder_Transformer(
        #     code_dim=512,
        #     embed_dim=512, # 1024
        #     output_dim=output_dim,
        #     block_size=49,
        #     num_layers=6,
        #     n_head=8
        # )
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        elif args.quantizer == "fsq":
            self.quantizer = FSQ(level, code_dim)



    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
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


    def forward(self, x):

        x_in = self.preprocess(x)
        # Encode
        # _x_in = x_in.reshape( int(x_in.shape[0]*4), x_in.shape[1], 16)
        # x_encoder = self.encoder(_x_in)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        # [Transformer Encoder]
        # _x_in = x_in.reshape( int(x_in.shape[0]*x_in.shape[2]/4), x_in.shape[1], 4)
        # _x_in = _x_in.permute(0,2,1)
        # x_encoder = self.encoder2(_x_in)
        # x_encoder = x_encoder.permute(0,2,1)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        x_encoder = self.encoder(x_in)
        # print('x_encoder', x_encoder.shape)
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        # print('x_quantized', x_quantized.shape)
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
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
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        output_dim = 56 * 6  # 55 joints, 1 trans, 2 person, 6d rotation
        self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)

        gender_emb_dim = 4
        shape_emb_dim = 40
        self.gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=gender_emb_dim)  # 2个类别: 0和1
        self.shape_projection = nn.Linear(10, shape_emb_dim)  # 10是你的体型向量维度
        # 3. 深度融合分支 (MLP)
        combined_input_dim = gender_emb_dim + shape_emb_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, int(code_dim/32))
        )

        self.decoder = Decoder(output_dim, output_emb_width+int(code_dim/32), down_t, stride_t, width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)


        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        elif args.quantizer == "fsq":
            self.quantizer = FSQ(level, code_dim)

        interx_skeleton = torch.from_numpy(INTERX_SKELETONS)
        self.fk_layers = FK_Single_Layer_Batch(
            SMPLX_TREE,
            interx_skeleton[:, 0],
            interx_skeleton[:, 1],
        )

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


        x_encoder = self.encoder(x_in)
        # print('x_encoder', x_encoder.shape)
        ## quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        # print('x_quantized', x_quantized.shape)
        ## decoder
        gender_embeded = self.gender_embedding(gender).reshape(-1, 4)
        shape_embeded = self.shape_projection(beta)
        combined = torch.cat([gender_embeded, shape_embeded], dim=1)
        condition_vector = self.condition_mlp(combined)
        condition_vector = condition_vector.unsqueeze(-1).expand(-1, -1, x_quantized.shape[-1])  # ([256, 32, 16])

        x_quantized = torch.cat([x_quantized, condition_vector], dim=1)

        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

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

class HumanVQVAE(nn.Module):
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
        self.vqvae = VQVAE_672_shape(args, nb_code, code_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, level=level)

    def forward(self, x, betas, gender, type='full'):
        '''type=[full, encode, decode]'''
        if type=='full':
            x_out, loss, perplexity = self.vqvae(x, betas, gender)
            return x_out, loss, perplexity
        elif type=='encode':
            b, t, c = x.size()
            quants = self.vqvae.encode(x) # (N, T)
            return quants
        elif type=='decode':
            x_out = self.vqvae.forward_decoder(x, betas, gender)
            return x_out
        else:
            raise ValueError(f'Unknown "{type}" type')
