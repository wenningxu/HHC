"""Official InterHuman evaluator path for paired HHC VQ reconstruction."""

from __future__ import annotations

import math

import numpy as np
import torch

from utils.runtime import evaluating, preserving_rng_state


@torch.no_grad()
def reconstruct_padded_interhuman_pair(model, motion1, motion2, lengths):
    """Reconstruct each real length without allowing padded frames into VQ."""
    if motion1.shape != motion2.shape or motion1.ndim != 3 or motion1.shape[-1] != 262:
        raise ValueError(f'expected paired [B,T,262], got {motion1.shape}, {motion2.shape}')
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=motion1.device)
    if lengths.shape != (motion1.shape[0],):
        raise ValueError(f'lengths must have shape {(motion1.shape[0],)}, got {lengths.shape}')
    if torch.any(lengths <= 0) or torch.any(lengths > motion1.shape[1]):
        raise ValueError('InterHuman reconstruction lengths are outside padded motion range')
    factor = int(getattr(model, 'temporal_downsample', 4))
    if factor <= 0:
        raise ValueError(f'invalid VQ temporal downsample {factor}')
    pair = torch.cat((motion1, motion2), dim=-1)
    reconstructed = torch.zeros_like(pair)
    for length in lengths.unique(sorted=True).tolist():
        selected = lengths == int(length)
        padded_length = int(math.ceil(int(length) / factor) * factor)
        if padded_length > pair.shape[1]:
            raise ValueError(
                f'length {length} needs {padded_length} frames but input has {pair.shape[1]}'
            )
        model_input = torch.zeros(
            int(selected.sum()), padded_length, 524,
            dtype=pair.dtype, device=pair.device,
        )
        model_input[:, :int(length)] = pair[selected, :int(length)]
        prediction, _, _ = model(model_input, type='full')
        if prediction.shape != model_input.shape:
            raise ValueError(
                f'InterHuman VQ reconstruction shape {prediction.shape} != {model_input.shape}'
            )
        reconstructed[selected, :int(length)] = prediction[:, :int(length)]
    return reconstructed[..., :262], reconstructed[..., 262:]


def reconstruction_errors(motion1, motion2, prediction1, prediction2, lengths, mean, std):
    """Return official-coordinate paired MPJPE and normalized feature MAE."""
    if not (motion1.shape == motion2.shape == prediction1.shape == prediction2.shape):
        raise ValueError('InterHuman reconstruction tensors must have identical shapes')
    if motion1.ndim != 3 or motion1.shape[-1] != 262:
        raise ValueError(f'expected paired [B,T,262], got {motion1.shape}')
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=motion1.device)
    frame_mask = torch.arange(motion1.shape[1], device=motion1.device)[None] < lengths[:, None]
    if not torch.any(frame_mask):
        raise ValueError('InterHuman reconstruction batch contains no valid frames')
    mean = torch.as_tensor(mean, dtype=motion1.dtype, device=motion1.device)
    std = torch.as_tensor(std, dtype=motion1.dtype, device=motion1.device)
    if mean.shape != (262,) or std.shape != (262,):
        raise ValueError(f'InterHuman stats must be 262D, got {mean.shape}, {std.shape}')
    joint_error_sum = motion1.new_zeros(())
    feature_error_sum = motion1.new_zeros(())
    joint_count = 0
    feature_count = 0
    for target, prediction in ((motion1, prediction1), (motion2, prediction2)):
        target_positions = (target * std + mean)[..., :66].reshape(*target.shape[:2], 22, 3)
        predicted_positions = (prediction * std + mean)[..., :66].reshape(
            *prediction.shape[:2], 22, 3
        )
        joint_error_sum += torch.linalg.vector_norm(
            predicted_positions - target_positions, dim=-1
        )[frame_mask].sum()
        joint_count += int(frame_mask.sum()) * 22
        absolute = torch.abs(prediction[..., :258] - target[..., :258])
        feature_error_sum += absolute[frame_mask].sum()
        feature_count += int(frame_mask.sum()) * 258
    return {
        'mpjpe_m': float((joint_error_sum / joint_count).item()),
        'normalized_feature_mae': float((feature_error_sum / feature_count).item()),
        'valid_person_frames': int(frame_mask.sum()) * 2,
    }


@torch.no_grad()
def evaluate_interhuman_vq(
    model,
    loader,
    evaluator,
    metrics_module,
    *,
    device='cuda:0',
    mean,
    std,
    random_seed=20260130,
):
    """Evaluate a paired VQ on the frozen 480-sample official val loader."""
    from tools.score_fair_interhuman_main import main_metrics_from_embeddings

    text_embeddings = []
    prediction_embeddings = []
    gt_embeddings = []
    weighted_mpjpe = 0.0
    weighted_mae = 0.0
    person_frames = 0
    batches = 0
    model_device = torch.device(device)
    with preserving_rng_state(), evaluating(model):
        for batch in loader:
            names, texts, motion1, motion2, lengths = batch
            motion1 = motion1.to(model_device).float()
            motion2 = motion2.to(model_device).float()
            prediction1, prediction2 = reconstruct_padded_interhuman_pair(
                model, motion1, motion2, lengths
            )
            prediction_batch = (names, texts, prediction1, prediction2, lengths)
            gt_batch = (names, texts, motion1, motion2, lengths)
            text, prediction = evaluator.get_co_embeddings(prediction_batch)
            gt = evaluator.get_motion_embeddings(gt_batch)
            text_embeddings.append(text.cpu().numpy())
            prediction_embeddings.append(prediction.cpu().numpy())
            gt_embeddings.append(gt.cpu().numpy())
            errors = reconstruction_errors(
                motion1, motion2, prediction1, prediction2, lengths, mean, std
            )
            weighted_mpjpe += errors['mpjpe_m'] * errors['valid_person_frames']
            weighted_mae += errors['normalized_feature_mae'] * errors['valid_person_frames']
            person_frames += errors['valid_person_frames']
            batches += 1
    if batches != 5 or person_frames <= 0:
        raise ValueError(
            f'official InterHuman val requires five complete batches, got {batches}'
        )
    metrics = main_metrics_from_embeddings(
        np.concatenate(text_embeddings),
        np.concatenate(prediction_embeddings),
        np.concatenate(gt_embeddings),
        metrics_module,
        batch_size=96,
        random_seed=random_seed,
    )
    metrics.update({
        'mpjpe_m': weighted_mpjpe / person_frames,
        'normalized_feature_mae': weighted_mae / person_frames,
        'valid_person_frames': person_frames,
    })
    return metrics
