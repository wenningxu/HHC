"""Anchor initial horizontal spacing to a paired GENERATED reference, not GT."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def align_initial_distance(motion, reference_first_frame, length):
    if (motion.ndim != 2 or motion.shape[1] != 672
            or reference_first_frame.shape != (672,) or not 0 < length <= len(motion)
            or not np.isfinite(motion[:length]).all()
            or not np.isfinite(reference_first_frame).all()):
        raise ValueError('finite raw672 motion/reference and valid length required')
    first, second = [660, 662], [666, 668]  # root x/z, person1/person2
    direction = motion[0, second].astype(np.float64) - motion[0, first]
    distance = np.linalg.norm(direction)
    target = np.linalg.norm(reference_first_frame[second].astype(np.float64)
                            - reference_first_frame[first])
    shift = direction * ((target - distance) / distance) if distance > 1e-8 else np.zeros(2)
    result = motion.copy()
    result[:length, second] += shift.astype(result.dtype)
    return result, shift, bool(distance <= 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.source, 'r') as source, h5py.File(args.reference, 'r') as reference:
        shape = source['motions'].shape
        if len(shape) != 4 or shape[2:] != (150, 672) or shape != reference['motions'].shape:
            raise ValueError('matching [samples,repeats,150,672] generations required')
        for f in (source, reference):
            if (f['completed_repeats'].shape != (shape[1],)
                    or not f['completed_repeats'][:].all()
                    or not f.attrs.get('checkpoint_sha256')):
                raise ValueError('complete model-generated inputs required; no GT reference')
        for key in ('names', 'lengths'):
            if not np.array_equal(source[key][:], reference[key][:]):
                raise ValueError(f'paired {key} mismatch')
        for key in ('requests_sha256', 'seed', 'checkpoint_sha256', 'vq_checkpoint_sha256'):
            if key not in source.attrs or source.attrs[key] != reference.attrs.get(key):
                raise ValueError(f'paired {key} mismatch')
        if source.attrs.get('beta_policy', 'requested') != reference.attrs.get('beta_policy', 'requested'):
            raise ValueError('paired beta policy mismatch')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shifts, skipped = [], 0
        with h5py.File(args.output, 'x') as target:
            target.attrs.update(dict(source.attrs))
            for key in ('save_token_ids', 'token_ids_layout'):
                if key in target.attrs:
                    del target.attrs[key]
            target.attrs.update({
                'diagnostic_not_formal_result': True,
                'sampler_semantics': 'diagnostic_two_generation_initial_horizontal_distance_anchor',
                'postprocess': 'person2_constant_xz_shift_to_generated_reference_initial_distance',
                'source_generation': str(args.source.resolve()),
                'source_generation_sha256': sha256(args.source),
                'reference_generation': str(args.reference.resolve()),
                'reference_generation_sha256': sha256(args.reference),
                'postprocess_source_sha256': sha256(Path(__file__)),
                'reference_cost_policy': 'requires a second model generation; not a free inference gain',
            })
            for key in ('names', 'lengths'):
                source.copy(key, target)
            completed = target.create_dataset('completed_repeats', data=np.zeros(shape[1], dtype=bool))
            motions = target.create_dataset('motions', shape=shape, dtype='float32',
                                            chunks=(1, 1, 150, 672), compression='lzf')
            for repeat in range(shape[1]):
                for index, length in enumerate(source['lengths'][:]):
                    adjusted, shift, was_skipped = align_initial_distance(
                        source['motions'][index, repeat],
                        reference['motions'][index, repeat, 0], int(length))
                    motions[index, repeat] = adjusted
                    shifts.append(float(np.linalg.norm(shift)))
                    skipped += was_skipped
                target.flush()
                completed[repeat] = True
                target.flush()
            target.attrs['mean_horizontal_shift_m'] = float(np.mean(shifts))
            target.attrs['max_horizontal_shift_m'] = float(np.max(shifts))
            target.attrs['degenerate_initial_directions_skipped'] = skipped
    print(json.dumps({'output': str(args.output), 'status': 'complete',
                      'mean_shift_m': float(np.mean(shifts)), 'skipped': skipped}))


if __name__ == '__main__':
    main()
