#!/usr/bin/env python
"""Convert the shared Inter-X source into CoShMDM's 22-body-joint format."""

import argparse
import os
import re
import shutil
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.fk import ForwardKinematicsLayerBatch
from utils.paramUtil import INTERX_SKELETONS, SMPLX_TREE
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d


def output_motion(rotations, translation, skeleton, fk):
    """Create the 312-D layout expected by CoShMDM's Inter-X loader."""
    rotation_6d = matrix_to_rotation_6d(axis_angle_to_matrix(rotations))
    offsets = skeleton.clone()
    offsets[1:] -= skeleton[SMPLX_TREE[1:]]
    coordinates, _ = fk(rotation_6d.unsqueeze(0), offsets.unsqueeze(0))
    coordinates = coordinates[0] + translation[:, None]

    # Upstream load_motion selects [0:22*3] positions and
    # [62*3:62*3+21*6] local body rotations.
    result = torch.zeros((len(rotations), 62 * 3 + 21 * 6), device=rotations.device)
    result[:, :22 * 3] = coordinates[:, :22].reshape(len(rotations), -1)
    result[:, 62 * 3:] = rotation_6d[:, 1:22].reshape(len(rotations), -1)
    return result.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default='dataset/interx')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    person1_dir = os.path.join(args.output_root, 'motions_processed', 'person1')
    person2_dir = os.path.join(args.output_root, 'motions_processed', 'person2')
    annotation_dir = os.path.join(args.output_root, 'annots')
    os.makedirs(person1_dir, exist_ok=True)
    os.makedirs(person2_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    for split in ('train', 'val', 'test'):
        shutil.copyfile(
            os.path.join(args.source_root, 'splits', f'{split}.txt'),
            os.path.join(args.output_root, f'{split}.txt'),
        )
    with open(os.path.join(args.source_root, 'splits', 'all.txt')) as handle:
        names = [line.strip() for line in handle if line.strip()]
    if args.limit is not None:
        names = names[:args.limit]

    device = torch.device(args.device)
    skeletons = torch.from_numpy(INTERX_SKELETONS).float().to(device)
    fk = ForwardKinematicsLayerBatch(SMPLX_TREE, device=device)
    written = skipped = 0
    with h5py.File(os.path.join(args.source_root, 'motions', 'all.h5'), 'r') as motions:
        for name in tqdm(names, desc='CoShMDM Inter-X conversion'):
            p1_path = os.path.join(person1_dir, name + '.npy')
            p2_path = os.path.join(person2_dir, name + '.npy')
            text_path = os.path.join(annotation_dir, name + '.txt')
            if not args.overwrite and all(os.path.isfile(path) for path in (p1_path, p2_path, text_path)):
                skipped += 1
                continue
            group_match = re.match(r'G(\d+)', name)
            if group_match is None:
                raise ValueError(f'cannot parse body-shape group from {name}')
            group_index = int(group_match.group(1)) - 1
            raw = torch.from_numpy(motions[name][:]).float().to(device)
            with torch.inference_mode():
                person1 = output_motion(raw[:, :55, :3], raw[:, 55, :3], skeletons[group_index, 0], fk)
                person2 = output_motion(raw[:, :55, 3:6], raw[:, 55, 3:6], skeletons[group_index, 1], fk)
            np.save(p1_path, person1)
            np.save(p2_path, person2)
            captions = []
            with open(os.path.join(args.source_root, 'texts_processed', name + '.txt')) as source:
                for line in source:
                    captions.append(line.split('#', 1)[0].strip())
            with open(text_path, 'w', encoding='utf-8') as target:
                target.write('\n'.join(captions) + '\n')
            written += 1

    print({
        'output_root': os.path.abspath(args.output_root),
        'requested': len(names),
        'written': written,
        'skipped': skipped,
        'motion_dim': 312,
        'splits': 'official Inter-X splits',
    })


if __name__ == '__main__':
    main()
