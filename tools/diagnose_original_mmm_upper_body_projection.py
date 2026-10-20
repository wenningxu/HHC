#!/usr/bin/env python
"""Project an original-MMM pilot back onto its input root/lower-body motion.

This is a diagnostic only.  It tests whether contact mismatch can improve while
foot motion remains exactly unchanged; it is not a replacement optimizer.
"""

import argparse
from pathlib import Path

import numpy as np


# SMPL-X joints whose rotations can affect the pelvis-to-foot chains.
PRESERVED_POSE_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)


def project(initial_path, optimized_path, output_path, translation_policy):
    with np.load(initial_path, allow_pickle=True) as handle:
        initial = {key: handle[key] for key in handle.files}
    with np.load(optimized_path, allow_pickle=True) as handle:
        optimized = {key: handle[key] for key in handle.files}
    poses = np.asarray(optimized['poses'], dtype=np.float32).copy()
    poses[:, PRESERVED_POSE_JOINTS] = np.asarray(
        initial['poses'], dtype=np.float32
    )[:, PRESERVED_POSE_JOINTS]
    payload = dict(optimized)
    payload['poses'] = poses
    initial_trans = np.asarray(initial['trans'], dtype=np.float32)
    optimized_trans = np.asarray(optimized['trans'], dtype=np.float32)
    if translation_policy == 'fixed':
        payload['trans'] = initial_trans.copy()
    elif translation_policy == 'constant_delta':
        delta = (optimized_trans - initial_trans).mean(axis=0, keepdims=True)
        payload['trans'] = initial_trans + delta
    else:
        raise ValueError(translation_policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-p1', required=True, type=Path)
    parser.add_argument('--initial-p2', required=True, type=Path)
    parser.add_argument('--optimized-p1', required=True, type=Path)
    parser.add_argument('--optimized-p2', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    for policy in ('fixed', 'constant_delta'):
        project(
            args.initial_p1, args.optimized_p1,
            args.output_dir / f'pilot_{policy}_p1.npz', policy,
        )
        project(
            args.initial_p2, args.optimized_p2,
            args.output_dir / f'pilot_{policy}_p2.npz', policy,
        )
    print(args.output_dir)


if __name__ == '__main__':
    main()
