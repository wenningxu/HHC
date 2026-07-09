"""Compatibility exports for two-person transformer models.

The active transformer implementations live in `models.two_person_transformer`.
Experimental copies are archived under `models.transformer_ablation`.
"""

from models.two_person_transformer import (
    Chi3DActionTransformer,
    Chi3DCondTransBase,
    CrossCondTransBase,
    CrossCondTransHead,
    Text2Motion_Transformer,
)

__all__ = [
    "Chi3DActionTransformer",
    "Chi3DCondTransBase",
    "CrossCondTransBase",
    "CrossCondTransHead",
    "Text2Motion_Transformer",
]
