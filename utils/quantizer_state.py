"""Checkpoint helpers for the non-buffer state in EMA vector quantizers.

``QuantizeEMAReset`` historically only registered ``codebook`` as a buffer.
Its running sums/counts and reset bookkeeping therefore need to be serialized
explicitly to make a training resume equivalent to uninterrupted training.
"""

from collections.abc import Mapping

import torch


QUANTIZER_NAMES = (
    "quantizer_upper",
    "quantizer_lower",
    "quantizer_root",
)


def _quantizer_names(module):
    if getattr(module, 'channel_names', None) == ('body', 'root'):
        return ('quantizer_upper', 'quantizer_root')
    return QUANTIZER_NAMES


def quantizer_training_state(module):
    """Return a CPU-portable snapshot of all EMA quantizer training state."""
    state = {}
    for name in _quantizer_names(module):
        quantizer = getattr(module, name)
        state[name] = {
            "init": bool(quantizer.init),
            "code_sum": (
                quantizer.code_sum.detach().clone()
                if quantizer.code_sum is not None
                else None
            ),
            "code_count": (
                quantizer.code_count.detach().clone()
                if quantizer.code_count is not None
                else None
            ),
            "reset_count": int(quantizer.reset_count),
            "usage": quantizer.usage.detach().clone(),
        }
    return state


def restore_quantizer_training_state(module, state):
    """Restore EMA state and report whether a legacy fallback was required.

    Old checkpoints have a valid codebook but omit the EMA accumulators.  The
    fallback deliberately marks the quantizer initialized, preserving that
    loaded codebook instead of letting the next forward pass overwrite it.
    """
    used_legacy_fallback = state is None
    if state is not None and not isinstance(state, Mapping):
        raise TypeError("quantizer_training_state must be a mapping or None")

    for name in _quantizer_names(module):
        quantizer = getattr(module, name)
        saved = None if state is None else state.get(name)
        if saved is None:
            used_legacy_fallback = True
            quantizer.init = True
            quantizer.code_sum = quantizer.codebook.detach().clone()
            quantizer.code_count = torch.ones(
                quantizer.nb_code, device=quantizer.codebook.device
            )
            quantizer.reset_count = 0
            quantizer.usage = torch.ones(
                quantizer.nb_code, 1, device=quantizer.codebook.device
            )
            continue

        quantizer.init = bool(saved["init"])
        quantizer.code_sum = (
            saved["code_sum"].to(quantizer.codebook.device)
            if saved["code_sum"] is not None
            else None
        )
        quantizer.code_count = (
            saved["code_count"].to(quantizer.codebook.device)
            if saved["code_count"] is not None
            else None
        )
        quantizer.reset_count = int(saved["reset_count"])
        quantizer.usage = saved["usage"].to(quantizer.codebook.device)

    return used_legacy_fallback
