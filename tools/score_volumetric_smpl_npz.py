#!/usr/bin/env python
"""Score NPZ motion pairs with a frozen, transparent VolumetricSMPL protocol."""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.calc_pene_from_npz import npz_to_amass_dict, to_torch_params
from optimization.volumetric_smpl_backend import VolumetricSMPLBackend


SCHEMA = 'hhc_volumetric_smpl_physical_metrics_v1'
SMPLX_BODY_SKELETON_JOINT_COUNT = 55


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def artifact_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def stable_sample_seed(base_seed, sample_key):
    suffix = int(hashlib.sha256(sample_key.encode('utf-8')).hexdigest()[:8], 16)
    return int((base_seed + suffix) % (2 ** 31 - 1))


def parse_sample_key(path, suffix='_pred_p1.npz'):
    if not path.name.endswith(suffix):
        raise ValueError(f'{path} does not end with {suffix}')
    sample_key = path.name[:-len(suffix)]
    match = re.fullmatch(r'(.+)_r(\d{3})', sample_key)
    if match:
        return sample_key, match.group(1), int(match.group(2))
    return sample_key, sample_key, 0


def discover_npz_pairs(folder):
    """Accept formal generation pairs and shape-augmentation runner pairs."""
    discovered = {}
    conventions = (
        ('*_pred_p1.npz', '_pred_p1.npz', '_pred_p2.npz', 'generated_prediction'),
        ('*_p1.npz', '_p1.npz', '_p2.npz', 'shape_augmentation'),
    )
    for pattern, p1_suffix, p2_suffix, convention in conventions:
        for p1_path in sorted(folder.glob(pattern)):
            if p1_suffix == '_p1.npz' and p1_path.name.endswith('_pred_p1.npz'):
                continue
            sample_key, sample_name, repeat_id = parse_sample_key(
                p1_path, p1_suffix
            )
            p2_path = folder / f'{sample_key}{p2_suffix}'
            if not p2_path.is_file():
                raise FileNotFoundError(f'missing paired motion: {p2_path}')
            if sample_key in discovered:
                raise ValueError(f'duplicate NPZ naming conventions for {sample_key}')
            discovered[sample_key] = (
                sample_key, sample_name, repeat_id, p1_path, p2_path,
                convention,
            )
    return [discovered[key] for key in sorted(discovered)]


def foot_sliding(joints):
    # SMPL-X body-joint indices; thresholds mirror the released optimizer.
    foot_indices = [7, 10, 8, 11]
    height_thresholds = joints.new_tensor([0.12, 0.05, 0.12, 0.05])
    if joints.shape[0] < 2:
        return {'contact_observations': 0, 'mean_horizontal_speed_m_per_frame': 0.0}
    horizontal_velocity = joints[1:, foot_indices][:, :, [0, 2]] - joints[:-1, foot_indices][:, :, [0, 2]]
    # Raw Inter-X translation is pelvis-centred. The released MMM optimization
    # entrypoint subtracts the sequence joint minimum before applying these
    # thresholds; reproduce that coordinate setting for the metric without
    # changing horizontal velocity or inter-person penetration geometry.
    # ``smplx`` appends face/hand landmark joints after the 55-joint SMPL-X
    # kinematic skeleton, while the optimizer's HBP BodyModel exposes only
    # that skeleton.  Contact optimization moves upper-body landmarks.  Using
    # every appended landmark to estimate the floor therefore changes the
    # grounded-foot mask even when root and the complete lower chain are
    # bitwise fixed.  Restrict the floor estimator to the shared skeleton so
    # optimization and scoring use the same coordinate contract.
    body_joint_count = min(
        int(joints.shape[1]), SMPLX_BODY_SKELETON_JOINT_COUNT
    )
    floor_y = joints[:, :body_joint_count, 1].min().detach()
    height = joints[:-1, foot_indices, 1] - floor_y
    grounded = height < height_thresholds
    speed = torch.linalg.vector_norm(horizontal_velocity, dim=-1)
    if not bool(grounded.any()):
        return {'contact_observations': 0, 'mean_horizontal_speed_m_per_frame': 0.0}
    return {
        'contact_observations': int(grounded.sum()),
        'mean_horizontal_speed_m_per_frame': float(speed[grounded].mean()),
    }


@torch.no_grad()
def score_pair(backend, params_a, params_b, seed):
    backend.begin_iteration(seed)
    output_a, gender_a = backend.body(params_a)
    output_b, gender_b = backend.body(params_b)
    frame_count = min(output_a.vertices.shape[0], output_b.vertices.shape[0])
    depth_sum = 0.0
    squared_sum = 0.0
    penetrating_count = 0
    collision_frames = 0
    max_depth = 0.0
    total_query_vertices = frame_count * (
        output_a.vertices.shape[1] + output_b.vertices.shape[1]
    )

    for frame_index in range(frame_count):
        frame_slice = slice(frame_index, frame_index + 1)
        sdf_b_to_a = backend._query_frame(
            output_b.vertices[frame_slice],
            backend._slice_output(output_a, frame_slice),
            gender_a,
        ).squeeze(0)
        sdf_a_to_b = backend._query_frame(
            output_a.vertices[frame_slice],
            backend._slice_output(output_b, frame_slice),
            gender_b,
        ).squeeze(0)
        depths = torch.cat((
            torch.relu(-sdf_b_to_a),
            torch.relu(-sdf_a_to_b),
        ))
        positive = depths > 0
        frame_count_positive = int(positive.sum())
        if frame_count_positive:
            collision_frames += 1
            active_depths = depths[positive]
            depth_sum += float(active_depths.sum())
            squared_sum += float((active_depths ** 2).sum())
            penetrating_count += frame_count_positive
            max_depth = max(max_depth, float(active_depths.max()))

    mean_active_depth = (
        depth_sum / penetrating_count if penetrating_count else 0.0
    )
    sliding_a = foot_sliding(output_a.joints)
    sliding_b = foot_sliding(output_b.joints)
    sliding_observations = (
        sliding_a['contact_observations'] + sliding_b['contact_observations']
    )
    mean_sliding = 0.0
    if sliding_observations:
        mean_sliding = (
            sliding_a['contact_observations']
            * sliding_a['mean_horizontal_speed_m_per_frame']
            + sliding_b['contact_observations']
            * sliding_b['mean_horizontal_speed_m_per_frame']
        ) / sliding_observations
    return {
        'frames': frame_count,
        'vertices_per_person': int(output_a.vertices.shape[1]),
        'bidirectional_query_vertices': int(total_query_vertices),
        'penetrating_vertex_count': penetrating_count,
        'penetrating_vertex_fraction': (
            penetrating_count / total_query_vertices
            if total_query_vertices else 0.0
        ),
        'collision_frame_count': collision_frames,
        'collision_frame_rate': collision_frames / frame_count if frame_count else 0.0,
        'penetration_depth_sum_m_per_frame': depth_sum / frame_count,
        'paper_eq6_squared_sum_m2_per_frame': squared_sum / frame_count,
        'mean_depth_m_on_penetrating_vertices': mean_active_depth,
        'max_depth_m': max_depth,
        'mean_foot_sliding_m_per_frame': mean_sliding,
        'foot_sliding_p1': sliding_a,
        'foot_sliding_p2': sliding_b,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--repeat-id', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    folder = args.folder.resolve()
    output_dir = args.output_dir.resolve()
    sample_dir = output_dir / 'samples'
    sample_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for pair in discover_npz_pairs(folder):
        sample_key, sample_name, repeat_id, p1_path, p2_path, convention = pair
        if args.repeat_id is not None and repeat_id != args.repeat_id:
            continue
        pairs.append(pair)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise FileNotFoundError('no NPZ motion pairs found')

    backend = VolumetricSMPLBackend(
        device=args.device,
        max_query_points_per_call=args.query_chunk,
    )
    repository = Path(__file__).resolve().parents[1]
    implementation = {
        'scorer': artifact_record(__file__),
        'volumetric_backend': artifact_record(
            repository / 'optimization/volumetric_smpl_backend.py'
        ),
        'npz_parameter_loader': artifact_record(
            repository / 'eval/calc_pene_from_npz.py'
        ),
    }
    metric_definitions = {
        'penetration_depth_sum_m_per_frame': 'mean over frames of bidirectional sum relu(-SDF), metres',
        'paper_eq6_squared_sum_m2_per_frame': 'mean over frames of bidirectional sum relu(-SDF)^2, square metres',
        'mean_depth_m_on_penetrating_vertices': 'bidirectional mean depth conditional on SDF<0, metres',
        'penetrating_vertex_fraction': 'fraction of all bidirectional queried vertices with SDF<0',
        'collision_frame_rate': 'fraction of frames containing at least one SDF<0 vertex',
        'max_depth_m': 'maximum bidirectional relu(-SDF), metres',
        'mean_foot_sliding_m_per_frame': 'grounded-foot mean horizontal displacement per frame, metres; floor is minimum y over the shared first 55 SMPL-X skeleton joints only',
    }
    scoring_contract = {
        'schema': SCHEMA,
        'device': str(torch.device(args.device)),
        'base_seed': args.seed,
        'sample_seed_rule': (
            'base + first32bits(sha256(sample_key)) mod 2^31-1'
        ),
        'max_frames': 150,
        'query_chunk': args.query_chunk,
        'foot_sliding_floor_policy': (
            'minimum_y_over_first_55_smplx_skeleton_joints; appended '
            'face/hand landmarks excluded'
        ),
        'volumetric_smpl_checkpoints': backend.checkpoint_records,
        'volumetric_smpl_representation': backend.representation_contract,
        'metric_definitions': metric_definitions,
        'implementation': implementation,
    }
    scoring_contract_sha256 = canonical_sha256(scoring_contract)
    records = []
    newly_scored_records = 0
    for sample_key, sample_name, repeat_id, p1_path, p2_path, convention in tqdm(
        pairs, desc='VolumetricSMPL physical metrics', unit='sample'
    ):
        record_path = sample_dir / f'{sample_key}.json'
        input_hashes = {
            'p1': sha256_file(p1_path),
            'p2': sha256_file(p2_path),
        }
        if record_path.is_file() and not args.overwrite:
            with open(record_path, encoding='utf-8') as handle:
                record = json.load(handle)
            if record.get('input_sha256') != input_hashes:
                raise ValueError(f'stale score artifact for {sample_key}')
            if record.get('scoring_contract_sha256') != scoring_contract_sha256:
                raise ValueError(
                    f'stale scoring contract for {sample_key}; use '
                    '--overwrite only for an intentional rescore'
                )
            if (
                record.get('schema') != SCHEMA
                or record.get('sample_key') != sample_key
                or record.get('sample_name') != sample_name
                or record.get('repeat_id') != repeat_id
                or Path(record.get('input_p1', '')).resolve()
                != p1_path.resolve()
                or Path(record.get('input_p2', '')).resolve()
                != p2_path.resolve()
                or record.get('seed')
                != stable_sample_seed(args.seed, sample_key)
            ):
                raise ValueError(f'stale score identity for {sample_key}')
            records.append(record)
            continue

        data_a = npz_to_amass_dict(dict(np.load(p1_path, allow_pickle=True)))
        data_b = npz_to_amass_dict(dict(np.load(p2_path, allow_pickle=True)))
        params_a = to_torch_params(data_a, args.device)
        params_b = to_torch_params(data_b, args.device)
        frame_count = min(params_a['trans'].shape[0], params_b['trans'].shape[0], 150)
        for params in (params_a, params_b):
            for key in ('root_orient', 'pose_body', 'pose_lhand', 'pose_rhand', 'trans'):
                params[key] = params[key][:frame_count]
        sample_seed = stable_sample_seed(args.seed, sample_key)
        metrics = score_pair(backend, params_a, params_b, sample_seed)
        record = {
            'schema': SCHEMA,
            'sample_key': sample_key,
            'sample_name': sample_name,
            'repeat_id': repeat_id,
            'input_naming_convention': convention,
            'input_p1': str(p1_path),
            'input_p2': str(p2_path),
            'input_sha256': input_hashes,
            'seed': sample_seed,
            'scoring_contract_sha256': scoring_contract_sha256,
            'metrics': metrics,
        }
        atomic_json(record_path, record)
        records.append(record)
        newly_scored_records += 1

    metric_keys = (
        'penetrating_vertex_fraction',
        'collision_frame_rate',
        'penetration_depth_sum_m_per_frame',
        'paper_eq6_squared_sum_m2_per_frame',
        'mean_depth_m_on_penetrating_vertices',
        'max_depth_m',
        'mean_foot_sliding_m_per_frame',
    )
    aggregate = {
        key: {
            'mean': float(np.mean([r['metrics'][key] for r in records])),
            'median': float(np.median([r['metrics'][key] for r in records])),
        }
        for key in metric_keys
    }
    expected_record_names = {f'{record["sample_key"]}.json' for record in records}
    stale_records = sorted(
        path.name for path in sample_dir.glob('*.json')
        if path.name not in expected_record_names
    )
    if stale_records:
        raise ValueError(
            f'output directory contains stale score records: '
            f'{stale_records[:5]}'
        )
    metrics_path = output_dir / 'metrics.json'
    created_at_utc = datetime.now(timezone.utc).isoformat()
    if not newly_scored_records and metrics_path.is_file():
        try:
            previous = json.loads(metrics_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if (
            previous.get('schema') == SCHEMA
            and previous.get('status') == 'complete'
            and previous.get('scoring_contract_sha256')
            == scoring_contract_sha256
            and previous.get('sample_count') == len(records)
            and previous.get('aggregate') == aggregate
        ):
            created_at_utc = previous.get('created_at_utc', created_at_utc)
    manifest = {
        'schema': SCHEMA,
        'status': 'complete',
        'created_at_utc': created_at_utc,
        'folder': str(folder),
        'accepted_input_naming_conventions': [
            '*_pred_p1.npz + *_pred_p2.npz',
            '*_p1.npz + *_p2.npz',
        ],
        'sample_count': len(records),
        'base_seed': args.seed,
        'sample_seed_rule': 'base + first32bits(sha256(sample_key)) mod 2^31-1',
        'max_frames': 150,
        'query_chunk': args.query_chunk,
        'device': args.device,
        'volumetric_smpl_checkpoints': backend.checkpoint_records,
        'volumetric_smpl_representation': backend.representation_contract,
        'metric_definitions': metric_definitions,
        'implementation': implementation,
        'scoring_contract': scoring_contract,
        'scoring_contract_sha256': scoring_contract_sha256,
        'aggregate': aggregate,
        'sample_records': [str((sample_dir / f'{r["sample_key"]}.json')) for r in records],
    }
    atomic_json(metrics_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
