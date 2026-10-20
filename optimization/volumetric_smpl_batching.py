"""Experimental exact frame batching for VolumetricSMPL Eq. 6 queries.

This module is deliberately separate from the frozen production backend while
the 25-sample serial pilot is running. It may only be promoted after value,
gradient, and CUDA-memory equivalence are measured on the same inputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _index_output(output, frame_indices):
    return SimpleNamespace(
        vertices=output.vertices.index_select(0, frame_indices),
        joints=output.joints.index_select(0, frame_indices),
        full_pose=output.full_pose.index_select(0, frame_indices),
    )


def _query_frames_preserving_serial_sampling(
    backend, query_points, target_output, target_gender
):
    """Batch neural evaluation while retaining serial per-frame mesh draws.

    VolumetricSMPL samples its target mesh stochastically inside
    ``encode_body``.  Calling that method on a frame batch changes the random
    draw assigned to each frame and therefore changes Eq. 6.  This helper
    performs the sampling-dependent body encoding frame by frame in the same
    order as production, then batches only the deterministic PointNet/implicit
    decoder work.  It remains experimental until value and gradient gates pass.
    """
    model = backend.models[backend.normalize_gender(target_gender)]
    volume = model.volume
    if not all(
        hasattr(volume, name)
        for name in (
            "compute_bone_trans", "get_bbox_bounds", "partitioner",
            "encoder", "query",
        )
    ):
        # Preserve the cheap analytic stand-in used to unit-test packing and
        # active-mask semantics without loading VolumetricSMPL weights.
        return backend._query_frame(query_points, target_output, target_gender)

    def query_function(points, vertices, joints, full_pose):
        bone_rows = []
        bbox_min_rows = []
        bbox_max_rows = []
        local_sample_rows = []
        for frame_index in range(points.shape[0]):
            frame_pose = full_pose[frame_index:frame_index + 1]
            frame_joints = joints[frame_index:frame_index + 1]
            frame_vertices = vertices[frame_index:frame_index + 1]
            bone_trans = volume.compute_bone_trans(frame_pose, frame_joints)
            bbox_min, bbox_max = volume.get_bbox_bounds(
                frame_vertices, bone_trans
            )
            sampled_points = volume.partitioner.sample_mesh_points(
                frame_vertices, volume.n_samples
            )
            batch, parts, samples, _ = sampled_points.shape
            local_samples = (
                bone_trans.reshape(batch, parts, 1, 4, 4).expand(
                    -1, -1, samples, -1, -1
                )
                @ F.pad(
                    sampled_points, [0, 1], "constant", 1.0
                ).unsqueeze(-1)
            )[..., :3, 0]
            bone_rows.append(bone_trans)
            bbox_min_rows.append(bbox_min)
            bbox_max_rows.append(bbox_max)
            local_sample_rows.append(local_samples)

        bone_trans = torch.cat(bone_rows, dim=0)
        bbox_min = torch.cat(bbox_min_rows, dim=0)
        bbox_max = torch.cat(bbox_max_rows, dim=0)
        local_samples = torch.cat(local_sample_rows, dim=0)
        batch, parts, samples, _ = local_samples.shape
        latent_code = volume.encoder(
            local_samples.reshape(batch * parts, samples, 3)
        ).view(batch, parts, -1)
        volume.impl_code = {
            "full_pose": full_pose,
            "joints": joints,
            "bone_trans": bone_trans,
            "latent_code": latent_code,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_size": torch.abs(bbox_max - bbox_min) * volume.bbox_padding,
            "bbox_center": (bbox_min + bbox_max) * 0.5,
        }
        return volume.query(points)

    inputs = (
        query_points,
        target_output.vertices,
        target_output.joints,
        target_output.full_pose,
    )
    volume.impl_code = None
    needs_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in inputs
    )
    if needs_grad:
        return checkpoint(
            query_function,
            *inputs,
            use_reentrant=True,
            preserve_rng_state=True,
        )
    return query_function(*inputs)


def penetration_squared_sum_frame_batched(
    backend,
    query_vertices,
    target_output,
    target_gender,
    temporal_mask,
    shallow_threshold=0.05,
    interaction_threshold=10.0,
    frame_batch_size=4,
    return_active_count=False,
):
    """Evaluate the unchanged active vertex set in small frame batches."""
    if query_vertices.ndim != 3 or query_vertices.shape[-1] != 3:
        raise ValueError('query vertices must have shape [T,V,3]')
    frame_count = query_vertices.shape[0]
    if target_output.vertices.shape[0] != frame_count:
        raise ValueError('query and target frame counts must match')
    if int(frame_batch_size) < 1:
        raise ValueError('frame_batch_size must be positive')
    if temporal_mask is None:
        temporal_mask = torch.zeros(
            frame_count,
            device=query_vertices.device,
            dtype=query_vertices.dtype,
        )
    else:
        temporal_mask = temporal_mask.reshape(-1)
        if temporal_mask.shape[0] != frame_count:
            raise ValueError('temporal mask and body frame counts must match')

    total = query_vertices.sum() * 0.0
    active_count = torch.zeros(
        (), dtype=torch.long, device=query_vertices.device
    )
    model = backend.models[backend.normalize_gender(target_gender)]
    padding_margin_ratio = (float(model.volume.bbox_padding) - 1.0) * 0.5

    for frame_start in range(0, frame_count, int(frame_batch_size)):
        frame_stop = min(frame_start + int(frame_batch_size), frame_count)
        candidate_rows = []
        selected_frames = []
        for frame_index in range(frame_start, frame_stop):
            query_frame = query_vertices[frame_index]
            target_vertices = target_output.vertices[frame_index].detach()
            raw_min = target_vertices.amin(dim=0)
            raw_max = target_vertices.amax(dim=0)
            margin = (raw_max - raw_min) * padding_margin_ratio
            bbox_min = raw_min - margin
            bbox_max = raw_max + margin
            candidate_mask = (
                (query_frame.detach() >= bbox_min)
                & (query_frame.detach() <= bbox_max)
            ).all(dim=-1)
            if bool(candidate_mask.any()):
                candidate_rows.append(query_frame[candidate_mask])
                selected_frames.append(frame_index)
        if not candidate_rows:
            continue

        max_candidates = max(row.shape[0] for row in candidate_rows)
        padded_rows = []
        valid_rows = []
        for row in candidate_rows:
            padding = max_candidates - row.shape[0]
            padded_rows.append(torch.cat([
                row,
                row.new_zeros((padding, 3)),
            ], dim=0))
            valid_rows.append(torch.cat([
                torch.ones(row.shape[0], dtype=torch.bool, device=row.device),
                torch.zeros(padding, dtype=torch.bool, device=row.device),
            ], dim=0))
        candidates = torch.stack(padded_rows, dim=0)
        candidate_valid = torch.stack(valid_rows, dim=0)
        frame_indices = torch.as_tensor(
            selected_frames,
            dtype=torch.long,
            device=query_vertices.device,
        )
        target_frames = _index_output(target_output, frame_indices)
        signed_distance = _query_frames_preserving_serial_sampling(
            backend,
            candidates, target_frames, target_gender
        )
        threshold = torch.where(
            temporal_mask.index_select(0, frame_indices) > 0.1,
            signed_distance.new_tensor(interaction_threshold),
            signed_distance.new_tensor(shallow_threshold),
        ).unsqueeze(1)
        active = (
            candidate_valid
            & (signed_distance < 0)
            & (signed_distance.abs() < threshold)
        )
        total = total + (signed_distance[active] ** 2).sum()
        active_count = active_count + active.sum()
    if return_active_count:
        return total, active_count
    return total
