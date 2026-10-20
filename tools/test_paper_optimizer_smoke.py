#!/usr/bin/env python
"""Short end-to-end smoke for the paper VolumetricSMPL optimizer path."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimize.optimize_full_body import optimize_full_body


def make_person(frames, gender, translation_x):
    return {
        'pose_body': np.zeros((frames, 63), dtype=np.float32),
        'pose_lhand': np.zeros((frames, 45), dtype=np.float32),
        'pose_rhand': np.zeros((frames, 45), dtype=np.float32),
        'root_orient': np.zeros((frames, 3), dtype=np.float32),
        'trans': np.tile(
            np.array([[translation_x, 0.0, 0.0]], dtype=np.float32),
            (frames, 1),
        ),
        'betas': np.zeros((1, 10), dtype=np.float32),
        'gender': gender,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument('--output')
    parser.add_argument('--motion-output-dir', type=Path)
    parser.add_argument('--semantic-contact', action='store_true')
    args = parser.parse_args()

    person_a = make_person(args.frames, 'male', 0.0)
    person_b = make_person(
        args.frames, 'female', 0.02 if args.semantic_contact else 0.15
    )
    temporal_mask = np.ones(args.frames, dtype=np.float32)
    interaction_pairs = None
    if args.semantic_contact:
        interaction_pairs = [{
            'p1_part': 'right hand (palm)',
            'p1_ids': [23, 26],
            'p2_part': 'left hand (palm)',
            'p2_ids': [17, 33],
            'contact_type': 'flat',
            'align_type': 'strict',
            'w_p1': 0.5,
            'w_p2': 0.5,
        }]
    final_a, final_b, diagnostics = optimize_full_body(
        person_a,
        person_b,
        temporal_mask=temporal_mask,
        interaction_pairs=interaction_pairs,
        num_epochs=args.epochs,
        freeze_epoch=max(0, args.epochs // 2),
        device=args.device,
        volumetric_smpl_query_chunk=args.query_chunk,
        return_diagnostics=True,
    )
    finite = all(
        np.isfinite(value).all()
        for person in (final_a, final_b)
        for value in person.values()
    )
    if not finite:
        raise RuntimeError('optimizer returned NaN or Inf')
    if diagnostics['signed_distance_backend'] != 'volumetric_smpl':
        raise RuntimeError('paper optimizer did not use VolumetricSMPL')
    report = {
        'device': args.device,
        'frames': args.frames,
        'epochs': args.epochs,
        'semantic_contact': args.semantic_contact,
        'finite_outputs': finite,
        'translation_delta_a': float(
            np.linalg.norm(final_a['trans'] - person_a['trans'])
        ),
        'translation_delta_b': float(
            np.linalg.norm(final_b['trans'] - person_b['trans'])
        ),
        'diagnostics': diagnostics,
    }
    if args.motion_output_dir:
        motion_output_dir = args.motion_output_dir.resolve()
        motion_output_dir.mkdir(parents=True, exist_ok=True)
        for person_index, (params, gender) in enumerate(
            ((final_a, 'male'), (final_b, 'female')), start=1
        ):
            poses = np.zeros((args.frames, 55, 3), dtype=np.float32)
            poses[:, 0] = params['root_orient'].reshape(-1, 3)
            poses[:, 1:22] = params['pose_body'].reshape(-1, 21, 3)
            poses[:, 25:40] = params['pose_lhand'].reshape(-1, 15, 3)
            poses[:, 40:55] = params['pose_rhand'].reshape(-1, 15, 3)
            target = motion_output_dir / f'smoke_r000_pred_p{person_index}.npz'
            temporary = target.with_suffix(target.suffix + f'.tmp.{os.getpid()}')
            with open(temporary, 'wb') as handle:
                np.savez_compressed(
                    handle,
                    poses=poses,
                    trans=params['trans'],
                    betas=params['betas'].reshape(-1)[:10],
                    gender=np.asarray(gender),
                    mocap_framerate=np.asarray(30),
                )
            os.replace(temporary, target)
        report['motion_output_dir'] = str(motion_output_dir)
    print(report)
    if args.output:
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f'.tmp.{os.getpid()}')
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        os.replace(temporary, target)


if __name__ == '__main__':
    main()
