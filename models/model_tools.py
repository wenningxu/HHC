from models.two_person_transformer import Text2Motion_Transformer
from models.interhuman_transformer import InterHumanText2MotionTransformer
from models.vqvae_sep3_shape import VQVAE_SEP3_shape
from models.vqvae_sep3_interhuman import VQVAE_SEP2_InterHuman


def build_model(args, model_type, **kwargs):
    if model_type == "interx_vq":
        return VQVAE_SEP3_shape(
            args,
            args.nb_code,
            args.code_dim,
            args.output_emb_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm,
        )
    if model_type == "interhuman_vq":
        return VQVAE_SEP2_InterHuman(
            args,
            args.nb_code,
            args.code_dim,
            args.output_emb_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm,
        )
    if model_type == "interx_transformer":
        return Text2Motion_Transformer(
            vqvae=kwargs["vqvae"],
            num_vq=args.nb_code,
            embed_dim=args.embed_dim_gpt,
            clip_dim=args.clip_dim,
            block_size=args.block_size,
            num_layers=args.num_layers,
            num_local_layer=args.num_local_layer,
            n_head=args.n_head_gpt,
            drop_out_rate=args.drop_out_rate,
            fc_rate=args.ff_rate,
            contact=args.contact_conditioning,
            shape=args.shape,
            paper_architecture=args.paper_architecture,
            conditioning_contract=args.conditioning_contract,
        )
    if model_type == "interhuman_transformer":
        if getattr(args, 'contact_conditioning', False):
            raise ValueError('InterHuman HHC v1 does not use Inter-X contact-token conditioning')
        if getattr(args, 'shape', False):
            raise ValueError('InterHuman HHC v1 has no beta/gender shape-conditioning inputs')
        return InterHumanText2MotionTransformer(
            vqvae=kwargs["vqvae"],
            num_vq=args.nb_code,
            embed_dim=args.embed_dim_gpt,
            clip_dim=args.clip_dim,
            block_size=args.block_size,
            num_layers=args.num_layers,
            num_local_layer=args.num_local_layer,
            n_head=args.n_head_gpt,
            drop_out_rate=args.drop_out_rate,
            fc_rate=args.ff_rate,
            contact=False,
            shape=False,
            paper_architecture=args.paper_architecture,
            conditioning_contract=args.conditioning_contract,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
