"""Frozen feature partition for adapting HHC to official InterHuman 262D."""

from __future__ import annotations

import numpy as np


JOINT_NAMES = (
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee',
    'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
    'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar',
    'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist',
)
JOINTS = len(JOINT_NAMES)
POSITION_SLICE = slice(0, JOINTS * 3)
VELOCITY_SLICE = slice(JOINTS * 3, JOINTS * 6)
ROTATION_SLICE = slice(JOINTS * 6, JOINTS * 6 + (JOINTS - 1) * 6)
FOOT_CONTACT_SLICE = slice(258, 262)
FEATURE_DIM = 262
PAIR_FEATURE_DIM = FEATURE_DIM * 2

ROOT_JOINTS = (0,)
BODY_JOINTS = tuple(
    index for index in range(JOINTS)
    if index not in ROOT_JOINTS
)


def feature_mask(joints, include_foot_contact=False):
    """Select position/velocity/rotation fields for one disjoint joint group."""
    joints = set(int(index) for index in joints)
    if not joints <= set(range(JOINTS)):
        raise ValueError(f'invalid InterHuman joint indices: {sorted(joints)}')
    mask = np.zeros(FEATURE_DIM, dtype=bool)
    for joint in joints:
        mask[joint * 3:(joint + 1) * 3] = True
        velocity = JOINTS * 3 + joint * 3
        mask[velocity:velocity + 3] = True
        if joint > 0:
            rotation = JOINTS * 6 + (joint - 1) * 6
            mask[rotation:rotation + 6] = True
    if include_foot_contact:
        mask[FOOT_CONTACT_SLICE] = True
    return mask


ROOT_MASK = feature_mask(ROOT_JOINTS)
BODY_MASK = feature_mask(BODY_JOINTS, include_foot_contact=True)
CHANNEL_MASKS = (BODY_MASK, ROOT_MASK)
CHANNEL_NAMES = ('body', 'root')
CASCADE_CHANNEL_ORDER = (1, 0)  # Root -> Body; arms belong to body.
PAIR_CHANNEL_MASKS = tuple(
    np.concatenate((mask, mask), axis=0) for mask in CHANNEL_MASKS
)


def validate_partition():
    stacked = np.stack(CHANNEL_MASKS, axis=0).astype(np.uint8)
    if not np.all(stacked.sum(axis=0) == 1):
        raise ValueError('InterHuman HHC feature masks are not disjoint/exhaustive')
    counts = tuple(int(mask.sum()) for mask in CHANNEL_MASKS)
    if counts != (256, 6):
        raise ValueError(f'unexpected InterHuman HHC channel dimensions: {counts}')
    pair_stacked = np.stack(PAIR_CHANNEL_MASKS, axis=0).astype(np.uint8)
    if pair_stacked.shape != (2, PAIR_FEATURE_DIM):
        raise ValueError(f'unexpected paired InterHuman mask shape: {pair_stacked.shape}')
    if not np.all(pair_stacked.sum(axis=0) == 1):
        raise ValueError('paired InterHuman HHC masks are not disjoint/exhaustive')
    pair_counts = tuple(int(mask.sum()) for mask in PAIR_CHANNEL_MASKS)
    if pair_counts != (512, 12):
        raise ValueError(f'unexpected paired InterHuman channel dimensions: {pair_counts}')
    return {
        'feature_dim': FEATURE_DIM,
        'pair_feature_dim': PAIR_FEATURE_DIM,
        'channel_order': CHANNEL_NAMES,
        'channel_dimensions': counts,
        'pair_channel_dimensions': pair_counts,
        'cascade_channel_order': CASCADE_CHANNEL_ORDER,
    }


validate_partition()
