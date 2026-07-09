"""Compatibility exports for legacy single-person VQ-VAE variants.

The active two-person VQ implementation lives in `models.vqvae_sep3_shape`.
"""

from models.vq_ablation.vqvae_legacy import HumanVQVAE, VQVAE_672, VQVAE_672_shape

__all__ = [
    "HumanVQVAE",
    "VQVAE_672",
    "VQVAE_672_shape",
]
