from models.two_person_transformer import Chi3DActionTransformer, Text2Motion_Transformer
from models.vqvae_sep3_shape import VQVAE_SEP3_Chi3DShape, VQVAE_SEP3_shape


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
    if model_type == "chi3d_vq":
        return VQVAE_SEP3_Chi3DShape(
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
            contact=args.contact,
            shape=args.shape,
        )
    if model_type == "chi3d_action_transformer":
        return Chi3DActionTransformer(
            vqvae=kwargs["vqvae"],
            num_actions=kwargs["num_actions"],
            num_groups=kwargs.get("num_groups", 3),
            num_vq=args.nb_code,
            embed_dim=args.embed_dim_gpt,
            block_size=args.block_size - 1,
            num_layers=args.num_layers,
            n_head=args.n_head_gpt,
            drop_out_rate=args.drop_out_rate,
            fc_rate=args.ff_rate,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
