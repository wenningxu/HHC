#!/usr/bin/env python
"""Convert strict fair-eval 672-D HDF5 generations to paired SMPL-X NPZ."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix


RESUME_CONTRACT = (
    'existing NPZ arrays must exactly equal the deterministic current '
    'request/generation conversion; missing files are atomically created; '
    'stale files fail closed'
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    os.replace(temporary, path)


def atomic_npz(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def validate_existing_npz(path, expected, sample_key):
    """Fail closed when resuming from an output written by another input."""
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(expected):
                raise ValueError(
                    f'{sample_key}: keys differ in existing {path.name}'
                )
            for key, expected_value in expected.items():
                actual = stored[key]
                expected_array = np.asarray(expected_value)
                if actual.dtype != expected_array.dtype:
                    raise ValueError(
                        f'{sample_key}: dtype differs for {path.name}:{key}'
                    )
                if actual.shape != expected_array.shape or not np.array_equal(
                    actual, expected_array
                ):
                    raise ValueError(
                        f'{sample_key}: values differ for {path.name}:{key}'
                    )
    except Exception as error:
        if isinstance(error, ValueError) and str(error).startswith(sample_key):
            raise
        raise ValueError(
            f'{sample_key}: cannot validate existing {path.name}: {error}'
        ) from error


def axis_angle(rotations_6d):
    # Float32 matrix->axis-angle is noticeably unstable close to pi (up to
    # ~3.7e-4 matrix error on the frozen GT smoke).  Convert in float64 and
    # store float32 parameters; the resulting round-trip matrix error is below
    # 2.4e-7 on that same sequence.
    tensor = torch.from_numpy(np.asarray(rotations_6d, dtype=np.float64))
    return matrix_to_axis_angle(rotation_6d_to_matrix(tensor)).float().numpy()


def person_payload(raw, person, betas, gender):
    start = 0 if person == 0 else 6
    rotations = axis_angle(raw[:, :55, start:start + 6])
    translation = raw[:, 55, start:start + 3].astype(np.float32)
    return {
        'poses': rotations.astype(np.float32),
        'trans': translation,
        'betas': np.asarray(betas, dtype=np.float32).reshape(10),
        'gender': np.asarray('male' if int(gender) == 0 else 'female'),
        'mocap_framerate': np.asarray(30),
    }


def capture_attrs(handle):
    result = {}
    for key, value in handle.attrs.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True, type=Path)
    parser.add_argument('--generations', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--repeat-id', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--resume', action='store_true',
        help=(
            'Reuse only existing per-person NPZ files whose arrays exactly '
            'match the current generation/request conversion. Missing files '
            'are created and stale files fail closed.'
        ),
    )
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error('--overwrite and --resume are mutually exclusive')

    requests_path = args.requests.resolve()
    generations_path = args.generations.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = np.load(requests_path, allow_pickle=False)
    names = request['names'].astype(str)
    captions = request['captions'].astype(str)
    lengths = request['lengths'].astype(np.int64)
    betas = request['betas'].astype(np.float32).reshape(-1, 2, 10)
    gender = request['gender'].astype(np.int64).reshape(-1, 2)
    if args.limit is not None:
        names = names[:args.limit]
        captions = captions[:args.limit]
        lengths = lengths[:args.limit]
        betas = betas[:args.limit]
        gender = gender[:args.limit]

    requests_sha256 = sha256_file(requests_path)
    generations_sha256 = sha256_file(generations_path)
    records = []
    with h5py.File(generations_path, 'r') as source:
        source_names = source['names'][:].astype(str)
        source_lengths = source['lengths'][:].astype(np.int64)
        if args.limit is not None:
            source_names = source_names[:args.limit]
            source_lengths = source_lengths[:args.limit]
        if not np.array_equal(source_names, names):
            raise ValueError('generation names/order differ from frozen requests')
        if not np.array_equal(source_lengths, lengths):
            raise ValueError('generation lengths differ from frozen requests')
        source_request_sha256 = source.attrs.get('requests_sha256')
        if source_request_sha256 is None:
            source_request_path = source.attrs.get('requests')
            if isinstance(source_request_path, bytes):
                source_request_path = source_request_path.decode('utf-8')
            if (
                not source_request_path
                or not os.path.isfile(source_request_path)
                or sha256_file(source_request_path) != requests_sha256
            ):
                raise ValueError(
                    'legacy generation HDF5 lacks a verifiable matching '
                    'request-file binding'
                )
        elif source_request_sha256 != requests_sha256:
            raise ValueError('generation HDF5 is bound to another request file')
        repeats = source['motions'].shape[1]
        if not 0 <= args.repeat_id < repeats:
            raise ValueError(f'repeat {args.repeat_id} is outside [0,{repeats})')
        if source['motions'].shape[0] < len(names) or source['motions'].shape[2:] != (150, 672):
            raise ValueError(f'unexpected generation shape {source["motions"].shape}')
        if 'completed_repeats' in source and not bool(
            source['completed_repeats'][args.repeat_id]
        ):
            raise ValueError('requested generation repeat is incomplete')
        hdf_attrs = capture_attrs(source)

        expected_files = set()
        for index, (name, caption, length) in enumerate(
            zip(names, captions, lengths)
        ):
            sample_key = f'{name}_r{args.repeat_id:03d}'
            p1_path = output_dir / f'{sample_key}_pred_p1.npz'
            p2_path = output_dir / f'{sample_key}_pred_p2.npz'
            expected_files.update((p1_path.name, p2_path.name))
            if (
                (p1_path.exists() or p2_path.exists())
                and not args.overwrite and not args.resume
            ):
                raise FileExistsError(
                    f'{sample_key}: output already exists; use --resume for '
                    'strict recovery or --overwrite only for an intentional '
                    'complete reconversion'
                )
            raw = source['motions'][index, args.repeat_id, :int(length)]
            if raw.shape != (int(length), 672) or not np.isfinite(raw).all():
                raise ValueError(f'{sample_key}: invalid generated motion')
            raw = raw.reshape(int(length), 56, 12)
            person_payloads = (
                person_payload(raw, 0, betas[index, 0], gender[index, 0]),
                person_payload(raw, 1, betas[index, 1], gender[index, 1]),
            )
            for person_path, payload in zip(
                (p1_path, p2_path), person_payloads
            ):
                if person_path.is_file() and args.resume:
                    validate_existing_npz(person_path, payload, sample_key)
                else:
                    atomic_npz(person_path, payload)
            records.append({
                'sample_name': name,
                'sample_key': sample_key,
                'repeat_id': args.repeat_id,
                'text': caption,
                'caption_sha256': hashlib.sha256(
                    caption.encode('utf-8')
                ).hexdigest(),
                'length': int(length),
                'request_index': index,
                'requests': str(requests_path),
                'requests_sha256': requests_sha256,
                'p1': str(p1_path),
                'p1_sha256': sha256_file(p1_path),
                'p2': str(p2_path),
                'p2_sha256': sha256_file(p2_path),
                'betas': betas[index].tolist(),
                'gender': gender[index].tolist(),
            })

    stale_npz = sorted(
        path.name for path in output_dir.glob('*_pred_p?.npz')
        if path.name not in expected_files
    )
    if stale_npz:
        raise ValueError(
            f'output directory contains stale paired NPZ files: {stale_npz[:5]}'
        )
    records_path = output_dir / 'generation_records.jsonl'
    atomic_jsonl(records_path, records)
    manifest = {
        'schema': 'hhc_fair_h5_to_optimizer_npz_v1',
        'requests': str(requests_path),
        'requests_sha256': requests_sha256,
        'generations': str(generations_path),
        'generations_sha256': generations_sha256,
        'generation_hdf5_attributes': hdf_attrs,
        'repeat_id': args.repeat_id,
        'samples': len(records),
        'generation_records': str(records_path),
        'generation_records_sha256': sha256_file(records_path),
        'rotation_conversion': '6D -> matrix -> axis-angle',
        'shape_gender_source': 'frozen request arrays',
        'mocap_framerate': 30,
        'resume_contract': RESUME_CONTRACT,
        'implementation': {
            'converter': artifact_record(__file__),
            'rotation_conversions': artifact_record(
                ROOT / 'utils/rotation_conversions.py'
            ),
        },
        'status': 'complete',
    }
    atomic_json(output_dir / '_conversion_manifest.json', manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
