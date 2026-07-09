"""Compatibility exports for VQ-VAE models.

The selected production VQ-VAE variants live in `models.vqvae_sep3_shape`.
Older ablation/experimental variants are archived under `models.vq_ablation`.
"""

from models.vqvae_sep3_shape import VQVAE_SEP3_Chi3DShape, VQVAE_SEP3_shape
from models.vq_ablation.vqvae_sep_ablation import (
    ConditionEncoder,
    VAE_CNN,
    VAE_SHAPE_CNN,
    VQVAE_SEP,
    VQVAE_SEP3,
    VQVAE_SEP3_shape_single,
    VQVAE_SEP_shape,
    VQVAE_SEP_shape_primitive,
    merge_body_hand,
    merge_upper_lower,
    upper_lower_sep,
)

__all__ = [
    "ConditionEncoder",
    "VAE_CNN",
    "VAE_SHAPE_CNN",
    "VQVAE_SEP",
    "VQVAE_SEP3",
    "VQVAE_SEP3_Chi3DShape",
    "VQVAE_SEP3_shape",
    "VQVAE_SEP3_shape_single",
    "VQVAE_SEP_shape",
    "VQVAE_SEP_shape_primitive",
    "merge_body_hand",
    "merge_upper_lower",
    "upper_lower_sep",
]
