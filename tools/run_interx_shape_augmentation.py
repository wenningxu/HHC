#!/usr/bin/env python
"""Optimize train motions after deterministic random Inter-X shape changes."""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.calc_pene_from_npz import npz_to_amass_dict, PeneCalculator, to_torch_params
from utils.interx_physical_ground import place_pair_on_mesh_ground, encode_physical_pair
from optimize.optimize_full_body import optimize_full_body
from optimization.volumetric_smpl_backend import VolumetricSMPLBackend
from tools.audit_interx_contact_pairs import validate_record
from tools.run_paper_contact_optimization import (
    original_mmm_diagnostics,
    output_payload,
)
from tools.test_data import _compute_geometry
from utils.interx_ground_alignment import (
    GROUND_ALIGNMENT_POLICY,
    SHARED_GROUND_ALIGNMENT_POLICY,
    align_pair_for_mmm_optimization,
    restore_person_to_raw_coordinates,
)
from utils.interx_shape_augmentation_plan import (
    validate_interx_shape_augmentation_plan,
)
from utils.paramUtil import CANDIDATES


PLAN_SCHEMA = 'hhc_interx_shape_augmentation_plan_v2'
PLAN_RECORD_SCHEMA = 'hhc_interx_shape_augmentation_record_v2'
CONTACT_SCHEMA = 'hhc_interx_contact_pairs_codex_strict_v3'
RUN_SCHEMA = 'hhc_interx_shape_augmentation_run_v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_npz(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(temporary, 'wb') as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('schema') != PLAN_RECORD_SCHEMA:
                raise ValueError(f'{path}:{line_number}: unsupported record schema')
            rows.append(row)
    names = [row['augmented_name'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('augmentation plan contains duplicate output names')
    return rows


def validate_plan(path):
    return validate_interx_shape_augmentation_plan(path)


def validate_contact_cache(path, split, require_full):
    path = path.resolve()
    manifest_path = path / '_extraction_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != CONTACT_SCHEMA or not manifest.get('strict_schema'):
        raise ValueError('contact cache is not the strict Codex v3 schema')
    if require_full and manifest.get('status') != 'complete':
        raise ValueError('formal augmentation requires a complete contact cache')
    if manifest.get('split_sha256') != sha256_file(split):
        raise ValueError('contact cache and augmentation plan use different splits')
    if require_full and (
        manifest.get('selection_offset') != 0
        or manifest.get('selection_limit') is not None
        or manifest.get('sample_count') != manifest.get('split_count')
        or manifest.get('completed_count') != manifest.get('sample_count')
    ):
        raise ValueError('formal augmentation requires a complete full-split cache')
    return path, manifest, manifest_path


def validate_caption_binding(metadata, caption, sample_name):
    """Bind plan/cache captions while tolerating boundary whitespace only."""
    metadata_caption = metadata.get('caption')
    if not isinstance(metadata_caption, str):
        raise ValueError(f'{sample_name}: contact metadata caption is not a string')
    metadata_caption_sha256 = hashlib.sha256(
        metadata_caption.encode('utf-8')
    ).hexdigest()
    if metadata.get('caption_sha256') != metadata_caption_sha256:
        raise ValueError(f'{sample_name}: contact metadata caption hash mismatch')
    if metadata_caption == caption:
        match_mode = 'exact'
    elif metadata_caption.strip() == caption.strip():
        match_mode = 'boundary_whitespace_normalized'
    else:
        raise ValueError(
            f'{sample_name}: plan/contact captions differ beyond boundary '
            'whitespace'
        )
    return {
        'match_mode': match_mode,
        'plan_caption_sha256': hashlib.sha256(
            caption.encode('utf-8')
        ).hexdigest(),
        'contact_caption_sha256': metadata_caption_sha256,
    }


def load_contacts(contact_dir, sample_name, caption, manifest):
    pair_path = contact_dir / f'{sample_name}.pkl'
    metadata_path = contact_dir / '_metadata' / f'{sample_name}.json'
    if not pair_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f'incomplete contact annotation for {sample_name}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    required = {
        'schema': CONTACT_SCHEMA,
        'sample_name': sample_name,
        'status': 'success',
        'provider': manifest['provider'],
        'requested_model': manifest['requested_model'],
        'reasoning_effort': manifest['reasoning_effort'],
        'prompt_template_sha256': manifest['prompt_template_sha256'],
    }
    mismatch = {
        key: (metadata.get(key), value) for key, value in required.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise ValueError(f'{sample_name}: contact metadata mismatch {mismatch}')
    caption_binding = validate_caption_binding(metadata, caption, sample_name)
    with open(pair_path, 'rb') as handle:
        contacts = pickle.load(handle)
    if not isinstance(contacts, list):
        raise ValueError(f'{sample_name}: contact payload is not a list')
    valid_rids = {
        int(region_id) for region_ids in CANDIDATES.values()
        for region_id in region_ids
    }
    errors = [
        error for contact in contacts
        for error in validate_record(contact, valid_rids)
    ]
    if errors:
        raise ValueError(f'{sample_name}: invalid contact rows {sorted(set(errors))}')
    return contacts, metadata, pair_path, metadata_path, caption_binding


def source_person_payload(raw, person, donor):
    channel = slice(0, 3) if person == 0 else slice(3, 6)
    return npz_to_amass_dict({
        'poses': raw[:, :55, channel].astype(np.float32),
        'trans': raw[:, 55, channel].astype(np.float32),
        'betas': np.asarray(donor['betas'], dtype=np.float32),
        'gender': np.asarray(donor['gender']),
    })


@torch.no_grad()
def prepare_physical_ground(full_raw, record, donors, calc):
    """Full-sequence actual mesh floors, then crop without framewise projection."""
    if len(full_raw) != record['source_frames']:
        raise ValueError('full source length differs from plan')
    originals = [{'betas': record['source_betas'][i], 'gender': record['source_genders'][i]}
                 for i in range(2)]
    result = []; all_floors = []
    for shapes in (originals, donors):
        floors = []
        for i in range(2):
            person = source_person_payload(full_raw, i, shapes[i])
            vertices, _, _ = calc._compute_geometry(to_torch_params(person, calc.device), compute_normals=False)
            floors.append(float(vertices[..., 1].min()))
        crop = full_raw[record['crop_start']:record['crop_stop']]
        result.append(place_pair_on_mesh_ground(
            [source_person_payload(crop, i, shapes[i]) for i in range(2)], floors))
        all_floors.append(floors)
    return result[0], result[1], all_floors


def load_timing(record):
    if record.get('timing_file') is None:
        raise ValueError(
            f'{record["source_name"]}: v2 augmentation requires an explicit '
            'manual timing file'
        )
    path = Path(record['timing_file']).resolve()
    if sha256_file(path) != record['timing_sha256']:
        raise ValueError(f'{record["source_name"]}: timing hash changed')
    timing = np.load(path, allow_pickle=False).astype(np.float32).reshape(-1)
    if len(timing) != record['source_frames']:
        raise ValueError(f'{record["source_name"]}: timing length changed')
    timing = timing[record['crop_start']:record['crop_stop']]
    if len(timing) != record['crop_frames'] or not set(np.unique(timing)) <= {0.0, 1.0}:
        raise ValueError(f'{record["source_name"]}: invalid cropped timing')
    if int(timing.sum()) != record['timing_positive_frames_crop']:
        raise ValueError(f'{record["source_name"]}: timing count differs from plan')
    return timing


def validate_completed(sidecar, out_p1, out_p2, expected):
    payload = json.loads(sidecar.read_text(encoding='utf-8'))
    mismatch = {
        key: (payload.get(key), value) for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f'{sidecar}: completed output mismatch {mismatch}')
    if payload.get('output_p1_sha256') != sha256_file(out_p1):
        raise ValueError(f'{out_p1}: output hash mismatch')
    if payload.get('output_p2_sha256') != sha256_file(out_p2):
        raise ValueError(f'{out_p2}: output hash mismatch')
    return payload


def sample_output_state(*paths):
    """Classify one sample's atomic files for restart-safe recovery."""
    present = [Path(path).is_file() for path in paths]
    if all(present):
        return 'complete'
    if any(present):
        return 'partial_recompute'
    return 'missing'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--plan', type=Path,
        default=Path(
            'configs/interx_shape_augmentation/timing_covered_seed123_v2/'
            'plan_manifest.json'
        ),
    )
    parser.add_argument(
        '--contact-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_codex_luna_strict_v3_train'),
    )
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument(
        '--mode', choices=('optimized', 'random_shape_only'),
        default='optimized',
        help='Main augmentation or the required no-optimization control.',
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--physical-mesh-ground', action='store_true',
                        help='Validated mesh-floor/reference preset with shared-origin output and static model reuse')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument(
        '--signed-distance-backend',
        choices=('nearest_vertex', 'volumetric_smpl'),
        default='nearest_vertex',
        help=(
            'Current MMM distance backend, or the single-factor '
            'VolumetricSMPL distance ablation.'
        ),
    )
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument(
        '--volumetric-smpl-penetration-vertex-count', type=int,
        help=(
            'Fixed seeded source-vertex sample for VolumetricSMPL '
            'penetration queries; omitted means all 10475 vertices.'
        ),
    )
    parser.add_argument(
        '--volumetric-smpl-body-encoding-samples', type=int,
        help='PointNet body samples per part; released default is 1000.',
    )
    parser.add_argument(
        '--volumetric-smpl-penetration-frame-count', type=int,
        help='Fixed-seed rotating frame sample for each penetration step.',
    )
    parser.add_argument(
        '--volumetric-smpl-frame-batch-size', type=int, default=1,
        help=(
            'Exact frame batching for VolumetricSMPL Eq. 6. A value of 1 '
            'preserves the serial production path.'
        ),
    )
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument(
        '--ground-alignment-policy',
        choices=(GROUND_ALIGNMENT_POLICY, SHARED_GROUND_ALIGNMENT_POLICY),
        default=GROUND_ALIGNMENT_POLICY,
        help=(
            'Default MMM per-person alignment, or a common pair translation '
            'that preserves relative geometry; restore offsets before export.'
        ),
    )
    parser.add_argument(
        '--foot-contact-mask-policy',
        choices=(
            'released_dynamic',
            'initial_or_current_grounded_horizontal_mean_speed',
        ),
        default='released_dynamic',
        help=(
            'Released per-iteration foot mask, or the preregistered '
            'single-factor iteration-0 frozen-mask ablation.'
        ),
    )
    parser.add_argument(
        '--contact-loss-reduction-policy',
        choices=('original_mmm',),
        default='original_mmm',
        help=(
            'Paper-literal temporal/pair sums, or the preregistered '
            'active-frame/contact-pair mean stabilization.'
        ),
    )
    parser.add_argument(
        '--optimization-variable-policy',
        choices=('released_all_per_frame',),
        default='released_all_per_frame',
        help=(
            'Released all-frame variables, or the preregistered local '
            'grounded-chain preservation policy.'
        ),
    )
    parser.add_argument('--fine-stage-start', type=int)
    parser.add_argument('--penetration-loss-reduction',
                        choices=('penetrating_vertex_mean', 'frame_sum'),
                        default='penetrating_vertex_mean')
    parser.add_argument('--loss-weights-json', default='{}')
    parser.add_argument('--aux-temporal-weighting',
                        choices=('released_broadcast', 'per_frame'),
                        default='released_broadcast')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument(
        '--selection-seed', type=int,
        help='Deterministically shuffle eligible plan rows before offset/limit.',
    )
    parser.add_argument('--positive-timing-only', action='store_true')
    parser.add_argument('--zero-timing-only', action='store_true')
    parser.add_argument(
        '--selection-position-base', type=int, default=0,
        help=(
            'Add this non-negative offset to completion-manifest selection '
            'positions. This permits disjoint timing-positive and timing-zero '
            'runs to materialize as one contiguous formal augmentation.'
        ),
    )
    parser.add_argument(
        '--allow-partial-contact-cache', action='store_true',
        help='Smoke tests only; formal augmentation must use the full train cache.',
    )
    parser.add_argument(
        '--preflight-only', action='store_true',
        help='Validate selected plan/contact/motion/timing inputs without GPU work.',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def select_record_shard(
        records, selection_seed=None, offset=0, limit=None,
        selection_position_base=0):
    """Select a restartable shard while retaining full-run positions."""
    ordered = list(records)
    if selection_seed is not None:
        selection_rng = np.random.default_rng(selection_seed)
        order = selection_rng.permutation(len(ordered))
        ordered = [ordered[int(index)] for index in order]
    if offset < 0 or selection_position_base < 0:
        raise ValueError('offset and selection-position-base must be non-negative')
    selected = list(enumerate(ordered))[offset:]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        selected = selected[:limit]
    return (
        [record for _, record in selected],
        [int(selection_position_base + position) for position, _ in selected],
    )


def main():
    args = parse_args()
    if args.physical_mesh_ground and not (
            args.mode == 'optimized' and args.num_epochs == 50 and args.learning_rate == .001
            and args.signed_distance_backend == 'nearest_vertex'
            and args.foot_contact_mask_policy == 'initial_or_current_grounded_horizontal_mean_speed'
            and args.penetration_loss_reduction == 'frame_sum'
            and args.aux_temporal_weighting == 'released_broadcast'
            and args.ground_alignment_policy == GROUND_ALIGNMENT_POLICY):
        raise ValueError('physical-mesh-ground requires validated 50-step/lr.001/nearest/frame_sum/metric-foot preset')
    if args.positive_timing_only and args.zero_timing_only:
        raise ValueError(
            '--positive-timing-only and --zero-timing-only are mutually exclusive'
        )
    if args.volumetric_smpl_frame_batch_size < 1:
        raise ValueError('--volumetric-smpl-frame-batch-size must be positive')
    plan, records_path, records, motion_file, split = validate_plan(args.plan)
    contact_dir, contact_manifest, contact_manifest_path = validate_contact_cache(
        args.contact_dir, split, require_full=not args.allow_partial_contact_cache,
    )
    if args.positive_timing_only:
        records = [row for row in records if row['timing_positive_frames_crop'] > 0]
    elif args.zero_timing_only:
        records = [row for row in records if row['timing_positive_frames_crop'] == 0]
    # Keep the position in the complete deterministic selection order.  This
    # position must be captured before offset/limit so a sample receives the
    # same optimization seed in a full run, a restart, or a process shard.
    records, selection_positions = select_record_shard(
        records,
        selection_seed=args.selection_seed,
        offset=args.offset,
        limit=args.limit,
        selection_position_base=args.selection_position_base,
    )
    if not records:
        raise ValueError('selected augmentation run is empty')
    loss_weights = json.loads(args.loss_weights_json)
    if not isinstance(loss_weights, dict):
        raise ValueError('loss-weights-json must decode to an object')
    fine_stage_start = 100 if args.fine_stage_start is None else args.fine_stage_start
    if fine_stage_start != 100:
        raise ValueError('original MMM hard-codes freeze_epoch=100')
    if loss_weights:
        raise ValueError('loss overrides are disabled for original-MMM numerics')
    if args.preflight_only:
        contact_rows = 0
        positive_timing = 0
        semantic_active = 0
        physical_timing_only = 0
        inactive_regions = 0
        boundary_whitespace_caption_matches = 0
        with h5py.File(motion_file, 'r') as motions:
            for record in records:
                raw = motions[record['source_name']][
                    record['crop_start']:record['crop_stop']
                ]
                if raw.shape != (record['crop_frames'], 56, 6):
                    raise ValueError(
                        f'{record["source_name"]}: cropped motion differs from plan'
                    )
                timing = load_timing(record)
                contacts, _, _, _, caption_binding = load_contacts(
                    contact_dir, record['source_name'], record['caption'],
                    contact_manifest,
                )
                boundary_whitespace_caption_matches += int(
                    caption_binding['match_mode']
                    == 'boundary_whitespace_normalized'
                )
                has_timing = bool(timing.sum() > 0)
                has_regions = bool(contacts)
                positive_timing += int(has_timing)
                semantic_active += int(has_timing and has_regions)
                physical_timing_only += int(has_timing and not has_regions)
                inactive_regions += int(not has_timing and has_regions)
                contact_rows += len(contacts)
        preflight = {
            'schema': 'hhc_interx_shape_augmentation_preflight_v1',
            'status': 'pass',
            'preflight': 'passed',
            'selected_samples': len(records),
            'positive_timing_samples': positive_timing,
            'semantic_contact_active_samples': semantic_active,
            'physical_timing_only_samples': physical_timing_only,
            'regions_inactive_because_timing_is_zero': inactive_regions,
            'structured_contact_rows': contact_rows,
            'boundary_whitespace_caption_matches': (
                boundary_whitespace_caption_matches
            ),
            'contact_cache_schema': contact_manifest['schema'],
            'fusion_policy': 'semantic attraction only on timing-positive frames',
            'caption_binding_policy': (
                'exact_utf8_or_boundary_whitespace_normalized; internal '
                'caption content must remain byte-identical'
            ),
            'inputs': {
                'plan': str(args.plan.resolve()),
                'plan_sha256': sha256_file(args.plan),
                'contact_manifest': str(contact_manifest_path),
                'contact_manifest_sha256': sha256_file(contact_manifest_path),
                'source_motion': str(motion_file),
                'source_motion_sha256': sha256_file(motion_file),
            },
            'implementation': {
                'path': str(Path(__file__).resolve()),
                'sha256': sha256_file(__file__),
            },
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        preflight_dir = args.output_dir.resolve()
        preflight_dir.mkdir(parents=True, exist_ok=True)
        preflight_path = preflight_dir / '_preflight_manifest.json'
        atomic_json(preflight_path, preflight)
        preflight['output'] = str(preflight_path)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_volumetric_smpl = None
    static_body_models = {} if args.physical_mesh_ground else None
    floor_calc = PeneCalculator('cpu') if args.physical_mesh_ground else None
    if args.signed_distance_backend == 'volumetric_smpl':
        shared_volumetric_smpl = VolumetricSMPLBackend(
            model_root='body_models',
            device=args.device,
            max_query_points_per_call=args.query_chunk,
            body_encoding_samples=args.volumetric_smpl_body_encoding_samples,
        )
    run_contract = {
        'schema': RUN_SCHEMA,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'plan': str(args.plan.resolve()),
        'plan_sha256': sha256_file(args.plan),
        'plan_records': str(records_path),
        'plan_records_sha256': sha256_file(records_path),
        'contact_manifest': str(contact_manifest_path),
        'contact_manifest_sha256': sha256_file(contact_manifest_path),
        'caption_binding_policy': (
            'exact_utf8_or_boundary_whitespace_normalized; internal caption '
            'content must remain byte-identical; both hashes recorded'
        ),
        'source_motion': str(motion_file),
        'source_motion_sha256': sha256_file(motion_file),
        'selection': {
            'offset': args.offset,
            'limit': args.limit,
            'positive_timing_only': args.positive_timing_only,
            'zero_timing_only': args.zero_timing_only,
            'selection_position_base': args.selection_position_base,
            'selection_seed': args.selection_seed,
            'selected_names': [row['augmented_name'] for row in records],
            'full_selection_positions': selection_positions,
        },
        'optimizer': {
            'implementation_contract': (
                'current_mmm_volumetric_distance_only'
                if args.signed_distance_backend == 'volumetric_smpl'
                else 'original_mmm_unmodified_numeric_path'
            ),
            'mode': args.mode,
            'iterations': args.num_epochs,
            'learning_rate': args.learning_rate,
            'fine_stage_start': fine_stage_start,
            'ground_alignment_policy': (
                args.ground_alignment_policy
                if args.mode == 'optimized' else 'not_applied_random_control'
            ),
            'ground_alignment_output_coordinates': (
                'temporary_mmm_floor_alignment_restored_to_interx_raw'
                if args.mode == 'optimized' else 'interx_raw_unchanged'
            ),
            'foot_contact_mask_policy': args.foot_contact_mask_policy,
            'penetration_loss_reduction': args.penetration_loss_reduction,
            'aux_temporal_weighting': args.aux_temporal_weighting,
            'contact_loss_reduction_policy': (
                args.contact_loss_reduction_policy
            ),
            'optimization_variable_policy': args.optimization_variable_policy,
            'loss_weight_overrides': loss_weights,
            'signed_distance_backend': args.signed_distance_backend,
            'signed_distance_backend_provenance': {
                'paper_contract': (
                    'Eq.4 and Eq.6 signed distances evaluated by the selected '
                    'backend; every nearest/index/reduction operation is fixed'
                ),
                'original_development_implementation': (
                    '/home/wenning/projects/MMM/optimization/losses.py'
                ),
                'original_development_implementation_sha256': sha256_file(
                    '/home/wenning/projects/MMM/optimization/losses.py'
                ),
                'original_implementation_operation': (
                    'CUDA Chamfer nearest vertex plus target vertex normal sign'
                ),
            },
            'volumetric_smpl_representation': (
                dict(shared_volumetric_smpl.representation_contract)
                if shared_volumetric_smpl is not None else {
                    'status': 'not_used',
                    'reason': 'unmodified original-MMM nearest-vertex optimizer',
                    'dataset': 'Inter-X',
                    'pose_layout': '55-joint axis-angle',
                    'hand_pose_convention': 'flat_hand_mean',
                    'smplx_flat_hand_mean': True,
                }
            ),
            'backend_lifetime': (
                'one_shared_instance_per_process'
                if shared_volumetric_smpl is not None else 'not_instantiated'
            ),
            'volumetric_smpl_checkpoints': (
                dict(shared_volumetric_smpl.checkpoint_records)
                if shared_volumetric_smpl is not None else {}
            ),
            'seed': args.seed,
            'seed_policy': (
                'base_seed_plus_100x_full_selection_position_before_slicing'
            ),
            'query_chunk': args.query_chunk,
            'volumetric_smpl_frame_batch_size': (
                args.volumetric_smpl_frame_batch_size
            ),
            'volumetric_smpl_penetration_vertex_count': (
                args.volumetric_smpl_penetration_vertex_count
            ),
            'volumetric_smpl_body_encoding_samples': (
                args.volumetric_smpl_body_encoding_samples
            ),
            'volumetric_smpl_penetration_frame_count': (
                args.volumetric_smpl_penetration_frame_count
            ),
            'temporal_mask': 'exact cropped manual binary timing',
            'contact_fusion': (
                'LLM regions define body parts; temporal timing gates semantic '
                'attraction; empty regions retain physical losses only'
            ),
        },
        'implementation': {
            str(path): sha256_file(path) for path in (
                Path(__file__).resolve(),
                ROOT / 'optimize/optimize_full_body.py',
                ROOT / 'optimization/losses.py',
                ROOT / 'utils/interx_ground_alignment.py',
            )
        },
        'requested_samples': len(records),
    }
    if args.foot_contact_mask_policy != 'released_dynamic':
        run_contract['optimizer']['implementation_contract'] = 'metric_aligned_foot_ablation_v1'
        run_contract['implementation'][str(ROOT / 'optimization/metric_aligned_foot_sliding.py')] = sha256_file(
            ROOT / 'optimization/metric_aligned_foot_sliding.py')
    if args.penetration_loss_reduction != 'penetrating_vertex_mean':
        run_contract['optimizer']['implementation_contract'] = 'penetration_reduction_ablation_v1'
    if args.aux_temporal_weighting != 'released_broadcast':
        run_contract['optimizer']['implementation_contract'] = 'aux_temporal_weighting_ablation_v1'
    if args.ground_alignment_policy != GROUND_ALIGNMENT_POLICY:
        run_contract['optimizer']['implementation_contract'] = 'shared_ground_alignment_ablation_v1'
    if args.physical_mesh_ground:
        run_contract['optimizer'].update({
            'implementation_contract': 'physical_mesh_ground_reference_v1',
            'ground_alignment_policy': 'full_sequence_actual_mesh_floor',
            'ground_alignment_output_coordinates': 'physical_pair_common_p1_origin',
            'ground_weight': 200., 'ground_temporal_weight': 1., 'ground_horizontal_weight': .5,
            'ground_support_reduction': 'height_decay', 'static_body_model_reuse': True,
            'seed_policy': 'fixed_base_seed_each_sample',
            'sample_timing_scope': 'source read through NPZ writes; excludes startup, sidecar/completion writes and scoring'})
        for path in ('utils/interx_physical_ground.py', 'optimization/mesh_ground_support.py', 'eval/calc_pene_from_npz.py'):
            run_contract['implementation'][str(ROOT / path)] = sha256_file(ROOT / path)
    run_contract['contract_sha256'] = canonical_sha256({
        key: value for key, value in run_contract.items()
        if key != 'created_at_utc'
    })
    contract_path = output_dir / '_run_contract.json'
    if contract_path.is_file() and not args.overwrite:
        previous = json.loads(contract_path.read_text(encoding='utf-8'))
        if previous.get('contract_sha256') != run_contract['contract_sha256']:
            raise ValueError('existing optimization directory has another contract')
        run_contract = previous
    else:
        atomic_json(contract_path, run_contract)

    completed = []
    measured_optimizer_seconds = []
    measured_processing_seconds = []
    with h5py.File(motion_file, 'r') as motions:
        for index, record in enumerate(records):
            source_name = record['source_name']
            augmented_name = record['augmented_name']
            selection_position = int(selection_positions[index])
            volumetric_smpl_seed = int(args.seed + selection_position * 100)
            if args.physical_mesh_ground:
                volumetric_smpl_seed = int(args.seed)
            sample_started = time.perf_counter()
            raw = motions[source_name][record['crop_start']:record['crop_stop']]
            if raw.shape != (record['crop_frames'], 56, 6):
                raise ValueError(f'{source_name}: cropped motion differs from plan')
            timing = load_timing(record)
            (
                contacts, contact_metadata, pair_path, metadata_path,
                caption_binding,
            ) = load_contacts(
                contact_dir, source_name, record['caption'], contact_manifest,
            )
            has_timing = bool(timing.sum() > 0)
            has_regions = bool(contacts)
            if has_timing and has_regions:
                contact_fusion_mode = 'timed_semantic_contact'
            elif has_timing:
                # The existing frame mask also captures geometric proximity
                # and incidental collision (for example self/object actions).
                # Do not hallucinate an inter-human body-part pair when the
                # caption contains none; penetration/regularization/ground and
                # skating losses still repair the random-shape motion.
                contact_fusion_mode = 'physical_timing_only_no_semantic_pair'
            elif has_regions:
                contact_fusion_mode = 'semantic_pair_inactive_zero_timing'
            else:
                contact_fusion_mode = 'no_timed_semantic_contact'
            out_p1 = output_dir / f'{augmented_name}_p1.npz'
            out_p2 = output_dir / f'{augmented_name}_p2.npz'
            sidecar = output_dir / f'{augmented_name}.json'
            identity = {
                'schema': 'hhc_interx_optimized_shape_augmentation_sample_v1',
                'source_name': source_name,
                'augmented_name': augmented_name,
                'run_contract_sha256': run_contract['contract_sha256'],
                'plan_record_sha256': canonical_sha256(record),
                'contact_pairs_sha256': sha256_file(pair_path),
                'contact_metadata_sha256': sha256_file(metadata_path),
                'contact_caption_match_mode': caption_binding['match_mode'],
                'plan_caption_sha256': caption_binding['plan_caption_sha256'],
                'contact_caption_sha256': (
                    caption_binding['contact_caption_sha256']
                ),
                'timing_sha256': record['timing_sha256'],
                'timing_positive_frames_crop': int(timing.sum()),
                'structured_contact_rows': len(contacts),
                'contact_fusion_mode': contact_fusion_mode,
                'crop_start': record['crop_start'],
                'crop_stop': record['crop_stop'],
                'selection_position': selection_position,
                'volumetric_smpl_seed': volumetric_smpl_seed,
            }
            output_state = sample_output_state(out_p1, out_p2, sidecar)
            if output_state == 'complete' and not args.overwrite:
                validate_completed(sidecar, out_p1, out_p2, identity)
                previous_sample = json.loads(sidecar.read_text(encoding='utf-8'))
                previous_diagnostics = previous_sample.get('diagnostics', {})
                if 'optimizer_wall_seconds' in previous_diagnostics:
                    measured_optimizer_seconds.append(float(
                        previous_diagnostics['optimizer_wall_seconds']
                    ))
                if 'sample_processing_wall_seconds' in previous_diagnostics:
                    measured_processing_seconds.append(float(
                        previous_diagnostics['sample_processing_wall_seconds']
                    ))
                completed.append(augmented_name)
                continue
            if output_state == 'partial_recompute' and not args.overwrite:
                # A host/process interruption can occur between the two atomic
                # NPZ replacements and the sidecar commit.  None of those
                # files is accepted as complete without the full hash-bound
                # triple; recomputing this one sample safely replaces partial
                # files while preserving every previously completed sample.
                print(
                    f'{augmented_name}: recovering interrupted partial output',
                    flush=True,
                )

            donors = sorted(record['shape_donors'], key=lambda row: row['target_person'])
            data_p1 = source_person_payload(raw, 0, donors[0])
            data_p2 = source_person_payload(raw, 1, donors[1])
            if args.mode == 'optimized':
                raw_data_p1 = data_p1
                raw_data_p2 = data_p2
                physical_reference = None
                if args.physical_mesh_ground:
                    original_people, physical_people, mesh_floors = prepare_physical_ground(
                        motions[source_name][:], record, donors, floor_calc)
                    data_p1, data_p2 = physical_people
                    physical_reference = [to_torch_params(p, args.device) for p in original_people]
                    torch.manual_seed(args.seed)
                else:
                    data_p1, floor_y_p1, data_p2, floor_y_p2 = align_pair_for_mmm_optimization(
                        raw_data_p1, raw_data_p2, _compute_geometry,
                        policy=args.ground_alignment_policy)
                timing_tensor = torch.as_tensor(
                    timing, dtype=torch.float32, device=args.device
                )
                if timing_tensor.is_cuda:
                    torch.cuda.synchronize(timing_tensor.device)
                optimizer_started = time.perf_counter()
                refined_p1, refined_p2 = optimize_full_body(
                    data_p1,
                    data_p2,
                    temporal_mask=timing_tensor,
                    interaction_pairs=contacts,
                    num_epochs=args.num_epochs,
                    lr=args.learning_rate,
                    device=args.device,
                    static_body_models=static_body_models,
                    physical_ground_reference=physical_reference,
                    signed_distance_backend=args.signed_distance_backend,
                    volumetric_smpl_query_chunk=args.query_chunk,
                    volumetric_smpl_frame_batch_size=(
                        args.volumetric_smpl_frame_batch_size
                    ),
                    volumetric_smpl_seed=volumetric_smpl_seed,
                    volumetric_smpl_penetration_vertex_count=(
                        args.volumetric_smpl_penetration_vertex_count
                    ),
                    volumetric_smpl_body_encoding_samples=(
                        args.volumetric_smpl_body_encoding_samples
                    ),
                    volumetric_smpl_penetration_frame_count=(
                        args.volumetric_smpl_penetration_frame_count
                    ),
                    volumetric_smpl_backend=shared_volumetric_smpl,
                    foot_contact_mask_policy=args.foot_contact_mask_policy,
                    penetration_loss_reduction=args.penetration_loss_reduction,
                    aux_temporal_weighting=args.aux_temporal_weighting,
                )
                if timing_tensor.is_cuda:
                    torch.cuda.synchronize(timing_tensor.device)
                optimizer_wall_seconds = time.perf_counter() - optimizer_started
                diagnostics = {
                    'implementation_contract': (
                        'current_mmm_volumetric_distance_only'
                        if args.signed_distance_backend == 'volumetric_smpl'
                        else 'original_mmm_unmodified_numeric_path'
                    ),
                    'signed_distance_backend': args.signed_distance_backend,
                    'volumetric_smpl_penetration_vertex_count': (
                        args.volumetric_smpl_penetration_vertex_count
                    ),
                    'volumetric_smpl_body_encoding_samples': (
                        args.volumetric_smpl_body_encoding_samples
                    ),
                    'volumetric_smpl_penetration_frame_count': (
                        args.volumetric_smpl_penetration_frame_count
                    ),
                    'iterations': int(args.num_epochs),
                    'learning_rate': float(args.learning_rate),
                    'foot_contact_mask_policy': args.foot_contact_mask_policy,
                    'penetration_loss_reduction': args.penetration_loss_reduction,
                    'aux_temporal_weighting': args.aux_temporal_weighting,
                    'fine_stage_start': 100,
                    'optimizer_wall_seconds': optimizer_wall_seconds,
                    'p1_changes_aligned_coordinates': original_mmm_diagnostics(
                        data_p1, refined_p1
                    ),
                    'p2_changes_aligned_coordinates': original_mmm_diagnostics(
                        data_p2, refined_p2
                    ),
                }
                if args.foot_contact_mask_policy != 'released_dynamic':
                    diagnostics['implementation_contract'] = 'metric_aligned_foot_ablation_v1'
                if args.penetration_loss_reduction != 'penetrating_vertex_mean':
                    diagnostics['implementation_contract'] = 'penetration_reduction_ablation_v1'
                if args.aux_temporal_weighting != 'released_broadcast':
                    diagnostics['implementation_contract'] = 'aux_temporal_weighting_ablation_v1'
                if args.ground_alignment_policy != GROUND_ALIGNMENT_POLICY:
                    diagnostics['implementation_contract'] = 'shared_ground_alignment_ablation_v1'
                if args.physical_mesh_ground:
                    (refined_p1, refined_p2), origin = encode_physical_pair([refined_p1, refined_p2])
                    diagnostics['implementation_contract'] = 'physical_mesh_ground_reference_v1'
                    diagnostics['ground_alignment'] = {
                        'policy': 'full_sequence_actual_mesh_floor',
                        'original_mesh_floors_m': mesh_floors[0], 'new_mesh_floors_m': mesh_floors[1],
                        'output_coordinates': 'physical_pair_common_p1_origin',
                        'common_export_origin': origin.tolist()}
                else:
                    refined_p1 = restore_person_to_raw_coordinates(refined_p1, floor_y_p1)
                    refined_p2 = restore_person_to_raw_coordinates(refined_p2, floor_y_p2)
                    diagnostics['ground_alignment'] = {
                        'policy': args.ground_alignment_policy,
                        'p1_floor_y_m': floor_y_p1, 'p2_floor_y_m': floor_y_p2,
                        'output_coordinates': 'restored_interx_raw',
                        'raw_root_y_mean_m': [float(np.mean(raw_data_p1['trans'][:, 1])),
                                              float(np.mean(raw_data_p2['trans'][:, 1]))],
                        'restored_root_y_mean_m': [float(np.mean(refined_p1['trans'][:, 1])),
                                                   float(np.mean(refined_p2['trans'][:, 1]))]}
            else:
                refined_p1, refined_p2 = data_p1, data_p2
                diagnostics = {
                    'mode': 'random_shape_only_no_optimization',
                    'iterations': 0,
                    'temporal_mask_supplied_to_optimizer': False,
                    'interaction_pairs_supplied_to_optimizer': False,
                }
            atomic_npz(out_p1, output_payload(refined_p1, donors[0]['gender']))
            atomic_npz(out_p2, output_payload(refined_p2, donors[1]['gender']))
            diagnostics['sample_processing_wall_seconds'] = (
                time.perf_counter() - sample_started
            )
            if args.mode == 'optimized':
                measured_optimizer_seconds.append(
                    float(diagnostics['optimizer_wall_seconds'])
                )
            measured_processing_seconds.append(
                float(diagnostics['sample_processing_wall_seconds'])
            )
            atomic_json(sidecar, {
                **identity,
                'caption': record['caption'],
                'timing_source': record['timing_source'],
                'source_shapes': {
                    'betas': record['source_betas'],
                    'genders': record['source_genders'],
                },
                'augmented_shapes': donors,
                'contact_annotation': {
                    'provider': contact_metadata['provider'],
                    'model': contact_metadata['requested_model'],
                    'session_id': contact_metadata['session_id'],
                },
                'diagnostics': diagnostics,
                'output_p1_sha256': sha256_file(out_p1),
                'output_p2_sha256': sha256_file(out_p2),
            })
            completed.append(augmented_name)
            atomic_json(output_dir / '_completion_manifest.json', {
                **run_contract,
                'completed_samples': len(completed),
                'completed_names': completed,
                'complete': False,
                'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            })

    runtime_summary = {
        'measured_samples': len(measured_processing_seconds),
        'sample_processing_wall_seconds_mean': float(np.mean(
            measured_processing_seconds
        )),
        'sample_processing_wall_seconds_median': float(np.median(
            measured_processing_seconds
        )),
        'sample_processing_wall_seconds_max': float(np.max(
            measured_processing_seconds
        )),
    }
    if measured_optimizer_seconds:
        runtime_summary.update({
            'optimizer_wall_seconds_measured_samples': len(
                measured_optimizer_seconds
            ),
            'optimizer_wall_seconds_mean': float(np.mean(
                measured_optimizer_seconds
            )),
            'optimizer_wall_seconds_median': float(np.median(
                measured_optimizer_seconds
            )),
            'optimizer_wall_seconds_max': float(np.max(
                measured_optimizer_seconds
            )),
            'projected_full_5868_optimizer_hours_single_gpu': float(
                np.mean(measured_optimizer_seconds) * 5868 / 3600
            ),
        })
    atomic_json(output_dir / '_completion_manifest.json', {
        **run_contract,
        'completed_samples': len(completed),
        'completed_names': completed,
        'complete': len(completed) == len(records),
        'runtime_summary': runtime_summary,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({
        'output_dir': str(output_dir),
        'completed': len(completed),
        'requested': len(records),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
