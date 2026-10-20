"""Coordinate transforms for MMM optimization on pelvis-centred Inter-X."""

from __future__ import annotations

import numpy as np


GROUND_ALIGNMENT_POLICY = "mmm_per_person_joint_min_restore_raw"
SHARED_GROUND_ALIGNMENT_POLICY = "pair_shared_joint_min_restore_raw"


def _copy_payload(payload):
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in payload.items()
    }


def align_person_for_mmm_optimization(payload, geometry_fn):
    """Apply the exact MMM pre-optimizer joint-min y alignment."""
    aligned = _copy_payload(payload)
    _vertices, joints = geometry_fn(aligned)
    floor_y = float(np.min(np.asarray(joints)[..., 1]))
    if not np.isfinite(floor_y):
        raise ValueError("non-finite SMPL-X joint floor")
    aligned["trans"] = np.asarray(aligned["trans"], dtype=np.float32).copy()
    aligned["trans"][:, 1] -= floor_y
    return aligned, floor_y


def restore_person_to_raw_coordinates(payload, floor_y):
    """Undo only the temporary alignment, preserving the optimized delta."""
    restored = _copy_payload(payload)
    floor_y = float(floor_y)
    if not np.isfinite(floor_y):
        raise ValueError("non-finite SMPL-X joint floor")
    restored["trans"] = np.asarray(
        restored["trans"], dtype=np.float32
    ).copy()
    restored["trans"][:, 1] += floor_y
    return restored


def align_pair_for_mmm_optimization(payload_a, payload_b, geometry_fn,
                                    policy=GROUND_ALIGNMENT_POLICY):
    """Optional common translation preserves all inter-person distances."""
    if policy not in (GROUND_ALIGNMENT_POLICY, SHARED_GROUND_ALIGNMENT_POLICY):
        raise ValueError(f"unsupported pair alignment policy: {policy}")
    a, floor_a = align_person_for_mmm_optimization(payload_a, geometry_fn)
    b, floor_b = align_person_for_mmm_optimization(payload_b, geometry_fn)
    if policy == SHARED_GROUND_ALIGNMENT_POLICY:
        shared_floor = min(floor_a, floor_b)
        # Reuse measured floors, but translate both original people identically.
        for aligned, original in ((a, payload_a), (b, payload_b)):
            aligned['trans'] = np.asarray(original['trans'], dtype=np.float32).copy()
            aligned['trans'][:, 1] -= shared_floor
        floor_a = floor_b = shared_floor
    return a, floor_a, b, floor_b
