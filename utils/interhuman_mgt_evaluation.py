"""Generation/evaluation helpers for the official InterHuman HHC contract."""

from __future__ import annotations

import random

import numpy as np
import torch

from utils.runtime import preserving_rng_state


def interhuman_token_lengths(processed_lengths, *, temporal_downsample=4):
    """Match released InterMask's integer division before MGT generation."""
    lengths = torch.as_tensor(processed_lengths, dtype=torch.long)
    factor = int(temporal_downsample)
    if factor <= 0:
        raise ValueError(f'invalid temporal downsample {factor}')
    if lengths.ndim != 1 or torch.any(lengths < 14) or torch.any(lengths > 299):
        raise ValueError(
            'official InterHuman processed lengths must be a vector in [14, 299]'
        )
    token_lengths = torch.div(lengths, factor, rounding_mode='floor')
    if torch.any(token_lengths <= 0):
        raise ValueError('InterHuman request has no valid VQ tokens')
    return token_lengths


@torch.no_grad()
def decode_interhuman_tokens(vq, tokens, token_lengths, *, padded_frames=300):
    """Decode only valid tokens and zero-pad like the released evaluator.

    HHC's fixed-width sampler returns special padding IDs outside the VQ
    vocabulary. They must never be sent to ``vq(..., type='decode')``. The
    official InterMask evaluator decodes the valid, floor-divided token span
    and pads the remaining frames with zeros; this function reproduces that
    behavior for the paired 524D HHC VQ.
    """
    if tokens.ndim != 3 or tokens.shape[-1] != 2:
        raise ValueError(f'expected [B,T,2] sampled tokens, got {tuple(tokens.shape)}')
    token_lengths = torch.as_tensor(
        token_lengths, dtype=torch.long, device=tokens.device
    )
    if token_lengths.shape != (tokens.shape[0],):
        raise ValueError(
            f'token_lengths must have shape {(tokens.shape[0],)}, '
            f'got {tuple(token_lengths.shape)}'
        )
    if torch.any(token_lengths <= 0) or torch.any(token_lengths > tokens.shape[1]):
        raise ValueError('token lengths are outside the sampled token sequence')
    factor = int(getattr(vq, 'temporal_downsample', 4))
    output = torch.zeros(
        tokens.shape[0], int(padded_frames), 524,
        dtype=torch.float32, device=tokens.device,
    )
    for length in token_lengths.unique(sorted=True).tolist():
        selected = token_lengths == int(length)
        valid_tokens = tokens[selected, :int(length)]
        code_count = int(getattr(vq, 'num_code', 0))
        if code_count and (
            torch.any(valid_tokens < 0) or torch.any(valid_tokens >= code_count)
        ):
            raise ValueError('valid InterHuman sample contains a non-code VQ token')
        decoded = vq(valid_tokens, type='decode')
        expected_frames = int(length) * factor
        expected_shape = (int(selected.sum()), expected_frames, 524)
        if tuple(decoded.shape) != expected_shape:
            raise ValueError(
                f'InterHuman VQ decode shape {tuple(decoded.shape)} != {expected_shape}'
            )
        if expected_frames > padded_frames:
            raise ValueError(
                f'decoded motion has {expected_frames} frames, exceeds {padded_frames}'
            )
        output[selected, :expected_frames] = decoded.float()
    return output[..., :262], output[..., 262:]


@torch.no_grad()
def generate_interhuman_pair(
    transformer,
    vq,
    clip_feature,
    word_embedding,
    processed_lengths,
    *,
    sampling_steps=50,
    temperature=0.5,
    cfg_scale=3.5,
    rand_pos=True,
    sample_seeds=None,
):
    """Generate normalized paired motions under the common-50 HHC sampler."""
    factor = int(getattr(vq, 'temporal_downsample', 4))
    token_lengths = interhuman_token_lengths(
        processed_lengths, temporal_downsample=factor
    ).to(clip_feature.device)
    if int(getattr(transformer, 'block_size', 0)) - 1 < int(token_lengths.max()):
        raise ValueError(
            'InterHuman transformer block_size cannot represent the request length'
        )
    # The unchanged HHC sampler internally applies ceil(m_length/factor).
    # Supplying the already-floored support as an exact multiple of ``factor``
    # reproduces InterMask while preserving the established Inter-X behavior.
    sample_arguments = {
        'm_length': token_lengths * factor,
        'max_steps': int(sampling_steps),
        'temperature': float(temperature),
        'CFG': float(cfg_scale),
        'rand_pos': bool(rand_pos),
        'type': 'sample_intra_cas',
    }
    if sample_seeds is not None:
        sample_arguments['sample_seeds'] = sample_seeds
    sampled = transformer(clip_feature, word_embedding, **sample_arguments)
    return decode_interhuman_tokens(vq, sampled, token_lengths)


@torch.no_grad()
def evaluate_interhuman_generator(
    generate_batch,
    loader,
    evaluator,
    metrics_module,
    *,
    random_seed=20260130,
    expected_batches=5,
    sample_seeds_by_name=None,
):
    """Score an in-memory generator on the frozen 480-sample val protocol.

    ``generate_batch(texts, processed_lengths)`` must return two normalized
    float tensors shaped like the padded official GT pair. Keeping text
    encoding and model-specific sampling behind this callback makes the
    evaluator contract independently testable and reusable by the future MGT
    trainer without copying InterCLIP metric equations.
    """
    from tools.score_fair_interhuman_main import main_metrics_from_embeddings

    text_embeddings = []
    prediction_embeddings = []
    gt_embeddings = []
    batches = 0
    with preserving_rng_state():
        random.seed(int(random_seed))
        np.random.seed(int(random_seed) % (2**32))
        torch.manual_seed(int(random_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(random_seed))
        for names, texts, motion1, motion2, lengths in loader:
            if sample_seeds_by_name is None:
                prediction1, prediction2 = generate_batch(list(texts), lengths)
            else:
                missing = [name for name in names if name not in sample_seeds_by_name]
                if missing:
                    raise ValueError(
                        f'validation seed map lacks request names: {missing[:5]}'
                    )
                prediction1, prediction2 = generate_batch(
                    list(texts), lengths,
                    [sample_seeds_by_name[name] for name in names],
                )
            if prediction1.shape != motion1.shape or prediction2.shape != motion2.shape:
                raise ValueError(
                    'InterHuman generator predictions must match padded GT shapes: '
                    f'{prediction1.shape}, {prediction2.shape}, {motion1.shape}'
                )
            prediction_batch = (
                names, texts, prediction1.cpu(), prediction2.cpu(), lengths,
            )
            gt_batch = (names, texts, motion1, motion2, lengths)
            text, prediction = evaluator.get_co_embeddings(prediction_batch)
            gt = evaluator.get_motion_embeddings(gt_batch)
            text_embeddings.append(text.cpu().numpy())
            prediction_embeddings.append(prediction.cpu().numpy())
            gt_embeddings.append(gt.cpu().numpy())
            batches += 1
    if batches != int(expected_batches):
        raise ValueError(
            f'official InterHuman validation requires {expected_batches} complete '
            f'batches, got {batches}'
        )
    result = main_metrics_from_embeddings(
        np.concatenate(text_embeddings),
        np.concatenate(prediction_embeddings),
        np.concatenate(gt_embeddings),
        metrics_module,
        batch_size=96,
        random_seed=int(random_seed),
    )
    non_finite = {
        key: value for key, value in result.items()
        if isinstance(value, (float, np.floating)) and not np.isfinite(value)
    }
    if non_finite:
        raise ValueError(f'non-finite InterHuman validation metrics: {non_finite}')
    return result


@torch.no_grad()
def evaluate_interhuman_validation_multimodality(
    generate_repeat,
    request_payload,
    evaluator,
    metrics_module,
    *,
    random_seed=20260132,
):
    """Compute a deterministic 100-prompt/11-repeat val-only diagnostic."""
    requests = request_payload.get('requests', [])
    repeats = int(request_payload.get('repeats', 0))
    pairs = int(request_payload.get('pairs', 0))
    if len(requests) != 100 or repeats != 11 or pairs != 10:
        raise ValueError(
            'InterHuman validation MModality requires 100 prompts, 11 repeats, 10 pairs'
        )
    names = [row['identifier'] for row in requests]
    texts = [row['caption'] for row in requests]
    lengths = torch.tensor(
        [row['processed_length'] for row in requests], dtype=torch.long
    )
    repeat_embeddings = []
    with preserving_rng_state():
        for repeat in range(repeats):
            repeat_seed = int(random_seed) + repeat
            random.seed(repeat_seed)
            np.random.seed(repeat_seed % (2**32))
            torch.manual_seed(repeat_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(repeat_seed)
            first, second = generate_repeat(texts, lengths, repeat)
            if first.shape != (100, 300, 262) or second.shape != first.shape:
                raise ValueError(
                    f'validation MM generator returned {first.shape}, {second.shape}'
                )
            embeddings = evaluator.get_motion_embeddings(
                (names, texts, first.cpu(), second.cpu(), lengths)
            )
            repeat_embeddings.append(embeddings.cpu().numpy())
        stacked = np.stack(repeat_embeddings, axis=1)
        np.random.seed(int(random_seed) % (2**32))
        value = metrics_module.calculate_multimodality(stacked, pairs, 6, 2)
    if not np.isfinite(value):
        raise ValueError(f'non-finite InterHuman validation MModality: {value}')
    return {
        'multimodality': float(value),
        'multimodality_prompts': 100,
        'multimodality_repeats': repeats,
        'multimodality_pairs': pairs,
    }
