"""Two-channel body/root VQ-VAE for paired InterHuman 262D representations.

This module is intentionally separate from ``vqvae_sep3_shape``.  The latter
is tied to the 672D Inter-X representation (or the unrelated 263D HumanML3D
fallback), includes SMPL-X FK and consumes beta/gender conditioning.  Official
InterHuman has none of those contracts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.encdec import Decoder, Encoder
from models.quantize_cnn import QuantizeEMAReset
from models.interhuman_quantizer import InterHumanQuantizeEMAReset
from utils.interhuman_representation import PAIR_CHANNEL_MASKS, PAIR_FEATURE_DIM


def _resolve_latent_dims(code_dim: int, configured=None) -> tuple[int, int]:
    """Return body/root widths, merging the former arm capacity into body."""
    if configured is None:
        body = code_dim * 3 // 4
        dims = (body, code_dim - body)
    else:
        if isinstance(configured, str):
            configured = configured.split(',')
        dims = tuple(int(value) for value in configured)
    if len(dims) != 2 or any(value <= 0 for value in dims):
        raise ValueError(f'InterHuman latent dims must be two positive integers: {dims}')
    if sum(dims) != int(code_dim):
        raise ValueError(
            f'InterHuman latent dims {dims} sum to {sum(dims)}, not code_dim={code_dim}'
        )
    return dims


class VQVAE_SEP2_InterHuman(nn.Module):
    """Separate body (including arms) and root quantizers, shared decoder.

    Token storage order is ``[body, root]``.  The MGT sampling
    cascade is a separate concern and consumes these channels in index order
    ``[1, 0]`` (Root -> Body).

    VQ input/output is ``[person1_262, person2_262]`` = 524D. ``beta`` and
    ``gender`` are accepted but ignored to preserve the transformer call
    signature; neither introduces shape conditioning.
    """

    feature_dim = PAIR_FEATURE_DIM
    channel_names = ('body', 'root')

    def __init__(
        self,
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
        sep_decoder=False,
    ):
        super().__init__()
        if sep_decoder:
            raise ValueError('InterHuman VQ supports only the shared 524D paired decoder')
        del output_emb_width, moment

        self.num_code = int(nb_code)
        self.code_dim = int(code_dim)
        self.temporal_downsample = int(stride_t) ** int(down_t)
        if self.temporal_downsample <= 0:
            raise ValueError('InterHuman VQ temporal downsample must be positive')
        self.latent_dims = _resolve_latent_dims(
            self.code_dim, getattr(args, 'interhuman_latent_dims', None)
        )

        # Copy makes registration independent from mutable NumPy globals.
        mask_tensor = torch.from_numpy(np.stack(PAIR_CHANNEL_MASKS).copy()).bool()
        if mask_tensor.shape != (2, PAIR_FEATURE_DIM):
            raise ValueError(f'unexpected InterHuman channel mask shape {mask_tensor.shape}')
        self.register_buffer('channel_masks', mask_tensor, persistent=True)
        input_dims = tuple(int(mask.sum()) for mask in PAIR_CHANNEL_MASKS)
        if input_dims != (512, 12):
            raise ValueError(f'unexpected InterHuman channel dimensions {input_dims}')

        encoder_kwargs = dict(
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.encoders = nn.ModuleList([
            Encoder(input_dim, latent_dim, **encoder_kwargs)
            for input_dim, latent_dim in zip(input_dims, self.latent_dims)
        ])
        quantizer_version = getattr(args, 'interhuman_quantizer', 'legacy')
        if quantizer_version not in ('legacy', 'fp32_window_reset_v1'):
            raise ValueError(f'unknown InterHuman quantizer: {quantizer_version}')
        quantizer_class = (InterHumanQuantizeEMAReset
                           if quantizer_version == 'fp32_window_reset_v1' else QuantizeEMAReset)
        self.quantizers = nn.ModuleList([
            quantizer_class(self.num_code, latent_dim, args)
            for latent_dim in self.latent_dims
        ])
        self.decoder = Decoder(
            PAIR_FEATURE_DIM,
            self.code_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

    # Named accessors do not register modules twice in the state dict.
    @property
    def quantizer_upper(self):
        return self.quantizers[0]

    @property
    def quantizer_root(self):
        return self.quantizers[1]

    @staticmethod
    def _preprocess(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float().contiguous()

    @staticmethod
    def _postprocess(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).contiguous()

    def _check_motion(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != PAIR_FEATURE_DIM:
            raise ValueError(
                f'InterHuman pair must be [batch,time,{PAIR_FEATURE_DIM}], got {tuple(x.shape)}'
            )
        if x.shape[1] % self.temporal_downsample:
            raise ValueError(
                f'InterHuman motion length {x.shape[1]} is not divisible by '
                f'temporal_downsample={self.temporal_downsample}'
            )

    def _encode_latents(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._check_motion(x)
        return [
            encoder(self._preprocess(x[..., mask]))
            for encoder, mask in zip(self.encoders, self.channel_masks)
        ]

    def forward(self, x, beta=None, gender=None, type='full'):
        del beta, gender
        if type == 'full':
            quantized = []
            losses = []
            perplexities = []
            for latent, quantizer in zip(self._encode_latents(x), self.quantizers):
                value, loss, perplexity = quantizer(latent)
                quantized.append(value)
                losses.append(loss)
                perplexities.append(perplexity)
            reconstructed = self._postprocess(self.decoder(torch.cat(quantized, dim=1)))
            if reconstructed.shape != x.shape:
                raise ValueError(
                    'VQ temporal reconstruction shape differs from input; use a '
                    f'window compatible with down_t/stride_t: {tuple(x.shape)} -> '
                    f'{tuple(reconstructed.shape)}'
                )
            return reconstructed, torch.stack(losses).sum(), torch.stack(perplexities)

        if type == 'encode':
            batch = x.shape[0]
            indices = []
            latent_lengths = set()
            for latent, quantizer in zip(self._encode_latents(x), self.quantizers):
                flattened = self._postprocess(latent).reshape(-1, latent.shape[1])
                code = quantizer.quantize(flattened).view(batch, -1)
                indices.append(code)
                latent_lengths.add(code.shape[1])
            if len(latent_lengths) != 1:
                raise ValueError(f'InterHuman VQ channel token lengths differ: {latent_lengths}')
            return torch.stack(indices, dim=-1)

        if type == 'decode':
            if x.ndim != 3 or x.shape[-1] != 2:
                raise ValueError(
                    f'InterHuman VQ tokens must be [batch,time,2], got {tuple(x.shape)}'
                )
            dequantized = [
                quantizer.dequantize(x[..., index])
                for index, quantizer in enumerate(self.quantizers)
            ]
            latent = torch.cat(dequantized, dim=-1).permute(0, 2, 1).contiguous()
            return self._postprocess(self.decoder(latent))

        raise ValueError(f'unknown InterHuman VQ forward type: {type!r}')


# Legacy import spelling only; old three-channel weights are incompatible.
VQVAE_SEP3_InterHuman = VQVAE_SEP2_InterHuman
