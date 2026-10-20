#!/usr/bin/env python
"""Run the unmodified original-MMM optimizer with strict provenance."""

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.calc_pene_from_npz import npz_to_amass_dict
from optimize.optimize_full_body import optimize_full_body
from tools.audit_interx_contact_pairs import validate_record
from utils.paramUtil import CANDIDATES


STRICT_SCHEMA = 'hhc_interx_contact_pairs_codex_strict_v3'


def synchronize_device(device):
    if not str(device).startswith('cuda'):
        return
    import torch
    torch.cuda.synchronize(device)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


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


def parse_sample_key(path):
    suffix = '_pred_p1.npz'
    if not path.name.endswith(suffix):
        raise ValueError(f'unexpected motion filename: {path.name}')
    sample_key = path.name[:-len(suffix)]
    match = re.fullmatch(r'(.+)_r(\d{3})', sample_key)
    if match:
        return sample_key, match.group(1), int(match.group(2))
    return sample_key, sample_key, 0


def load_generation_records(path):
    if path is None:
        return {}
    records = {}
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record['sample_name'], int(record['repeat_id']))
            if key in records:
                raise ValueError(
                    f'duplicate generation record {key} at line {line_number}'
                )
            records[key] = record
    return records


def validate_generation_record_inputs(
    record, p1_path, p2_path, sample_key, sample_name, repeat_id,
):
    """Bind each optimizer input byte-for-byte to its conversion record."""
    p1_path = p1_path.resolve()
    p2_path = p2_path.resolve()
    p1_sha256 = sha256_file(p1_path)
    p2_sha256 = sha256_file(p2_path)
    expected = {
        'sample_key': sample_key,
        'sample_name': sample_name,
        'repeat_id': int(repeat_id),
        'p1': str(p1_path),
        'p1_sha256': p1_sha256,
        'p2': str(p2_path),
        'p2_sha256': p2_sha256,
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual = record.get(key)
        if key in {'p1', 'p2'} and isinstance(actual, str):
            actual = str(Path(actual).resolve())
        if actual != expected_value:
            mismatches[key] = {
                'actual': actual,
                'expected': expected_value,
            }
    if mismatches:
        raise ValueError(
            f'{sample_name}: generation record does not bind optimizer '
            f'inputs: {json.dumps(mismatches, sort_keys=True)}'
        )
    return p1_sha256, p2_sha256


def validate_contact_cache(contact_dir, expected_split):
    manifest_path = contact_dir / '_extraction_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'strict extraction manifest is missing: {manifest_path}'
        )
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('schema') != STRICT_SCHEMA or not manifest.get('strict_schema'):
        raise ValueError('contact cache is not the strict versioned schema')
    expected_split_hash = sha256_file(expected_split)
    if manifest.get('split_sha256') != expected_split_hash:
        raise ValueError('contact-cache split differs from the requested split')
    if manifest.get('status') != 'complete':
        raise ValueError('strict contact cache extraction is not complete')
    if manifest.get('completed_count') != manifest.get('sample_count'):
        raise ValueError('strict contact cache completion count is inconsistent')
    if manifest.get('caption_source') == 'frozen_fair_evaluation_requests':
        for key in (
            'requests', 'requests_sha256', 'request_names_sha256',
            'request_captions_sha256', 'contract_sha256',
        ):
            if not manifest.get(key):
                raise ValueError(
                    f'strict frozen-request contact cache lacks {key}'
                )
        requests_path = Path(manifest['requests']).resolve()
        if not requests_path.is_file():
            raise FileNotFoundError(
                f'frozen contact requests are missing: {requests_path}'
            )
        if sha256_file(requests_path) != manifest['requests_sha256']:
            raise ValueError('frozen contact request hash has changed')
        with np.load(requests_path, allow_pickle=False) as request:
            names = request['names'].astype(str).tolist()
            captions = request['captions'].astype(str).tolist()
        if (len(names) != manifest['sample_count']
                or canonical_sha256(names)
                != manifest['request_names_sha256']
                or canonical_sha256(captions)
                != manifest['request_captions_sha256']):
            raise ValueError(
                'frozen contact request names/captions differ from manifest'
            )
    return manifest, manifest_path, expected_split_hash


def load_strict_contacts(contact_dir, sample_name, extraction_manifest):
    pair_path = contact_dir / f'{sample_name}.pkl'
    metadata_path = contact_dir / '_metadata' / f'{sample_name}.json'
    if not pair_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f'incomplete strict contact record for {sample_name}'
        )
    with open(metadata_path, encoding='utf-8') as handle:
        metadata = json.load(handle)
    required = {
        'schema': STRICT_SCHEMA,
        'sample_name': sample_name,
        'status': 'success',
        'prompt_template_sha256': extraction_manifest[
            'prompt_template_sha256'
        ],
        'provider': 'openai_codex_cli_chatgpt_login',
        'requested_model': extraction_manifest['requested_model'],
        'reasoning_effort': extraction_manifest['reasoning_effort'],
        'response_model_version': extraction_manifest['requested_model'],
    }
    if extraction_manifest.get('caption_source') == (
        'frozen_fair_evaluation_requests'
    ):
        required.update({
            'caption_source': 'frozen_fair_evaluation_requests',
            'requests_sha256': extraction_manifest['requests_sha256'],
        })
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f'{sample_name} contact metadata {key!r} mismatch: '
                f'{metadata.get(key)!r} vs {expected!r}'
            )
    if extraction_manifest.get('caption_source') == (
        'frozen_fair_evaluation_requests'
    ):
        writer = metadata.get('request_binding_writer')
        if not isinstance(writer, dict):
            raise ValueError(
                f'{sample_name} contact metadata lacks request binding writer'
            )
        writer_path = Path(str(writer.get('path', ''))).resolve()
        writer_hash = writer.get('sha256')
        if (
            not writer_path.is_file()
            or not isinstance(writer_hash, str)
            or sha256_file(writer_path) != writer_hash
        ):
            raise ValueError(
                f'{sample_name} request binding writer is missing or changed'
            )
    with open(pair_path, 'rb') as handle:
        contacts = pickle.load(handle)
    if not isinstance(contacts, list):
        raise ValueError(f'{sample_name} contact payload is not a list')
    valid_region_ids = {
        int(region_id)
        for region_ids in CANDIDATES.values()
        for region_id in region_ids
    }
    errors = [
        error
        for record in contacts
        for error in validate_record(record, valid_region_ids)
    ]
    if errors:
        raise ValueError(
            f'{sample_name} contact payload failed strict validation: '
            f'{sorted(set(errors))}'
        )
    return contacts, metadata, pair_path, metadata_path


def output_payload(params, original_gender):
    poses = np.zeros((params['trans'].shape[0], 55, 3), dtype=np.float32)
    poses[:, 0] = params['root_orient'].reshape(-1, 3)
    poses[:, 1:22] = params['pose_body'].reshape(-1, 21, 3)
    poses[:, 25:40] = params['pose_lhand'].reshape(-1, 15, 3)
    poses[:, 40:55] = params['pose_rhand'].reshape(-1, 15, 3)
    return {
        'poses': poses,
        'trans': params['trans'].astype(np.float32),
        'betas': params['betas'].reshape(-1)[:10].astype(np.float32),
        'gender': np.asarray(original_gender),
        'mocap_framerate': np.asarray(30),
    }


def original_mmm_diagnostics(initial, refined):
    """Measure changes externally without modifying the original optimizer."""
    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    diagnostics = {}
    for key in ('root_orient', 'trans', 'pose_body', 'pose_lhand', 'pose_rhand'):
        delta = array(refined[key]) - array(initial[key])
        diagnostics[key] = {
            'max_abs_delta': float(np.max(np.abs(delta))),
            'rms_delta': float(np.sqrt(np.mean(delta * delta))),
        }
    trans_delta = array(refined['trans']) - array(initial['trans'])
    diagnostics['trans']['temporal_max_abs_deviation'] = float(np.max(np.abs(
        trans_delta - trans_delta.mean(axis=0, keepdims=True)
    )))
    return diagnostics


def validate_completed_output(
    sidecar_path,
    out_p1,
    out_p2,
    expected,
):
    """Reject stale/corrupt outputs instead of silently treating them as done."""
    with open(sidecar_path, encoding='utf-8') as handle:
        sidecar = json.load(handle)
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise ValueError(
                f'{sidecar_path}: completed-output {key!r} mismatch: '
                f'{sidecar.get(key)!r} vs {value!r}'
            )
    output_hashes = {
        'output_p1_sha256': sha256_file(out_p1),
        'output_p2_sha256': sha256_file(out_p2),
    }
    for key, value in output_hashes.items():
        if sidecar.get(key) != value:
            raise ValueError(
                f'{sidecar_path}: output hash mismatch for {key}'
            )
    if sidecar.get('schema') != 'hhc_paper_contact_optimization_sample_v1':
        raise ValueError(f'{sidecar_path}: unsupported sample schema')
    runtime = sidecar.get('optimization_wall_seconds')
    if not isinstance(runtime, (int, float)) or not np.isfinite(runtime) \
            or runtime < 0:
        raise ValueError(
            f'{sidecar_path}: missing/invalid optimization wall time'
        )
    return sidecar


def sample_output_state(*paths):
    """Classify a hash-bound output triple for safe interrupted resume."""
    present = [Path(path).is_file() for path in paths]
    if all(present):
        return 'complete'
    if any(present):
        return 'partial_recompute'
    return 'missing'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-dir', type=Path, required=True)
    parser.add_argument('--generation-manifest', type=Path)
    parser.add_argument(
        '--contact-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_codex_luna_strict_v3_test'),
    )
    parser.add_argument(
        '--expected-split', type=Path,
        default=Path('dataset/interx/splits/test.txt'),
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repeat-id', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--query-chunk', type=int, default=512)
    parser.add_argument(
        '--signed-distance-backend', choices=('nearest_vertex',),
        default='nearest_vertex',
        help='Fixed original-MMM CUDA Chamfer nearest-vertex backend.',
    )
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--fine-stage-start', type=int, default=100)
    parser.add_argument(
        '--loss-weights-json', default='{}',
        help='Must be empty for the unmodified original-MMM implementation.',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    motion_dir = args.motion_dir.resolve()
    contact_dir = args.contact_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = [
        line.strip() for line in args.expected_split.read_text().splitlines()
        if line.strip()
    ]
    expected_set = set(expected_names)
    extraction_manifest, extraction_manifest_path, split_sha256 = (
        validate_contact_cache(contact_dir, args.expected_split)
    )
    generation_records = load_generation_records(args.generation_manifest)
    loss_weights = json.loads(args.loss_weights_json)
    if not isinstance(loss_weights, dict):
        raise ValueError('--loss-weights-json must decode to an object')
    if loss_weights:
        raise ValueError(
            'loss overrides are disabled: formal runs use byte-identical '
            'original-MMM optimization/losses.py'
        )
    if args.fine_stage_start != 100:
        raise ValueError(
            'original MMM hard-codes freeze_epoch=100; '
            '--fine-stage-start must equal 100'
        )
    fine_stage_start = 100
    p1_paths = sorted(motion_dir.glob('*_pred_p1.npz'))
    candidates = []
    for p1_path in p1_paths:
        sample_key, sample_name, repeat_id = parse_sample_key(p1_path)
        if repeat_id == args.repeat_id and sample_name in expected_set:
            candidates.append((p1_path, sample_key, sample_name, repeat_id))
    if args.limit is not None:
        candidates = candidates[:args.limit]
    if not candidates:
        raise FileNotFoundError('no matching generated motion pairs were found')

    run_contract = {
        'schema': 'hhc_paper_contact_optimization_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'motion_dir': str(motion_dir),
        'generation_manifest': (
            str(args.generation_manifest.resolve())
            if args.generation_manifest else None
        ),
        'generation_manifest_sha256': (
            sha256_file(args.generation_manifest)
            if args.generation_manifest else None
        ),
        'contact_dir': str(contact_dir),
        'contact_extraction_manifest': str(extraction_manifest_path),
        'contact_extraction_manifest_sha256': sha256_file(
            extraction_manifest_path
        ),
        'expected_split': str(args.expected_split.resolve()),
        'expected_split_sha256': split_sha256,
        'expected_split_count': len(expected_names),
        'repeat_id': args.repeat_id,
        'optimizer': {
            'implementation_contract': 'original_mmm_unmodified_numeric_path',
            'iterations': args.num_epochs,
            'learning_rate': args.learning_rate,
            'fine_stage_start': fine_stage_start,
            'loss_weight_overrides': loss_weights,
            'signed_distance_backend': args.signed_distance_backend,
            'signed_distance_backend_provenance': {
                'paper_contract': (
                    'Eq.4 and Eq.6 require signed distance but do not mandate '
                    'a VolumetricSMPL implementation'
                ),
                'original_development_implementation': str(
                    Path('/home/wenning/projects/MMM/optimization/losses.py')
                ),
                'original_development_implementation_sha256': sha256_file(
                    '/home/wenning/projects/MMM/optimization/losses.py'
                ),
                'original_operation': (
                    'CUDA Chamfer nearest vertex plus target vertex normal sign'
                ),
            },
            'seed_role': 'provenance_only_original_optimizer_is_deterministic',
            'query_chunk_role': 'unused_by_original_optimizer',
        },
        'requested_samples': len(candidates),
        'requested_sample_keys': [item[1] for item in candidates],
        'implementation': {
            'runner': {
                'path': str(Path(__file__).resolve()),
                'sha256': sha256_file(__file__),
            },
            'optimizer': {
                'path': str((ROOT / 'optimize/optimize_full_body.py').resolve()),
                'sha256': sha256_file(ROOT / 'optimize/optimize_full_body.py'),
            },
            'losses': {
                'path': str((ROOT / 'optimization/losses.py').resolve()),
                'sha256': sha256_file(ROOT / 'optimization/losses.py'),
            },
            'original_mmm_losses': {
                'path': '/home/wenning/projects/MMM/optimization/losses.py',
                'sha256': sha256_file(
                    '/home/wenning/projects/MMM/optimization/losses.py'
                ),
            },
        },
    }
    contract_identity = {
        key: value for key, value in run_contract.items()
        if key != 'created_at_utc'
    }
    run_contract['contract_sha256'] = canonical_sha256(contract_identity)
    contract_path = output_dir / '_run_contract.json'
    if contract_path.is_file() and not args.overwrite:
        with open(contract_path, encoding='utf-8') as handle:
            previous_contract = json.load(handle)
        if previous_contract.get('contract_sha256') != run_contract['contract_sha256']:
            raise ValueError(
                'resume contract differs from the existing optimization run; '
                'use a new output directory or explicitly --overwrite'
            )
        # Preserve the original creation timestamp on an exact resume.
        run_contract = previous_contract
    else:
        atomic_json(contract_path, run_contract)

    invocation_start = time.perf_counter()
    completed = []
    optimization_wall_seconds = []
    for index, (p1_path, sample_key, sample_name, repeat_id) in enumerate(
        candidates
    ):
        p2_path = motion_dir / f'{sample_key}_pred_p2.npz'
        if not p2_path.is_file():
            raise FileNotFoundError(f'missing paired motion: {p2_path}')
        out_p1 = output_dir / f'{sample_key}_pred_p1.npz'
        out_p2 = output_dir / f'{sample_key}_pred_p2.npz'
        sidecar = output_dir / f'{sample_key}.json'
        contacts, contact_metadata, pair_path, metadata_path = (
            load_strict_contacts(
                contact_dir, sample_name, extraction_manifest
            )
        )
        generation_record = generation_records.get((sample_name, repeat_id))
        if args.generation_manifest and generation_record is None:
            raise ValueError(
                f'generation manifest lacks {(sample_name, repeat_id)}'
            )
        if generation_record is not None:
            input_p1_sha256, input_p2_sha256 = (
                validate_generation_record_inputs(
                    generation_record, p1_path, p2_path, sample_key,
                    sample_name, repeat_id,
                )
            )
            if generation_record.get('text') != contact_metadata.get('caption'):
                raise ValueError(
                    f'{sample_name} generation caption differs from the '
                    'strict LLM contact caption'
                )
            expected_caption_hash = sha256_text(generation_record['text'])
            if generation_record.get('caption_sha256') != expected_caption_hash:
                raise ValueError(
                    f'{sample_name} generation caption hash is missing or invalid'
                )
            if contact_metadata.get('caption_sha256') != expected_caption_hash:
                raise ValueError(
                    f'{sample_name} contact caption hash is missing or invalid'
                )
            if extraction_manifest.get('caption_source') == (
                'frozen_fair_evaluation_requests'
            ):
                request_hash = extraction_manifest['requests_sha256']
                if generation_record.get('requests_sha256') != request_hash:
                    raise ValueError(
                        f'{sample_name} generation and contact extraction use '
                        'different frozen requests'
                    )
                if generation_record.get('request_index') != (
                    contact_metadata.get('request_index')
                ):
                    raise ValueError(
                        f'{sample_name} frozen request index differs between '
                        'generation and contact extraction'
                    )
        else:
            input_p1_sha256 = sha256_file(p1_path)
            input_p2_sha256 = sha256_file(p2_path)

        generation_record_sha256 = (
            canonical_sha256(generation_record)
            if generation_record is not None else None
        )
        contact_pairs_sha256 = sha256_file(pair_path)
        contact_metadata_sha256 = sha256_file(metadata_path)
        completed_identity = {
            'sample_key': sample_key,
            'sample_name': sample_name,
            'repeat_id': repeat_id,
            'run_contract_sha256': run_contract['contract_sha256'],
            'input_p1': str(p1_path.resolve()),
            'input_p1_sha256': input_p1_sha256,
            'input_p2': str(p2_path.resolve()),
            'input_p2_sha256': input_p2_sha256,
            'contact_pairs': str(pair_path),
            'contact_pairs_sha256': contact_pairs_sha256,
            'contact_metadata': str(metadata_path),
            'contact_metadata_sha256': contact_metadata_sha256,
            'generation_record_sha256': generation_record_sha256,
            'caption': contact_metadata['caption'],
            'structured_contact_rows': len(contacts),
        }
        output_state = sample_output_state(out_p1, out_p2, sidecar)
        if output_state == 'complete' and not args.overwrite:
            completed_sidecar = validate_completed_output(
                sidecar, out_p1, out_p2, completed_identity
            )
            completed.append(sample_key)
            optimization_wall_seconds.append(float(
                completed_sidecar['optimization_wall_seconds']
            ))
            continue
        if output_state == 'partial_recompute' and not args.overwrite:
            # Each file is atomically replaced, but a host interruption can
            # occur between the two NPZ commits and the final sidecar.  The
            # incomplete triple is never accepted as evidence; recompute just
            # this sample and atomically replace its remnants.
            print(
                f'{sample_key}: recovering interrupted partial optimization',
                flush=True,
            )

        data_p1 = npz_to_amass_dict(
            dict(np.load(p1_path, allow_pickle=True))
        )
        data_p2 = npz_to_amass_dict(
            dict(np.load(p2_path, allow_pickle=True))
        )
        original_genders = (data_p1['gender'], data_p2['gender'])
        synchronize_device(args.device)
        optimization_start = time.perf_counter()
        refined_p1, refined_p2 = optimize_full_body(
            data_p1,
            data_p2,
            interaction_pairs=contacts,
            num_epochs=args.num_epochs,
            lr=args.learning_rate,
            device=args.device,
        )
        diagnostics = {
            'implementation_contract': 'original_mmm_unmodified_numeric_path',
            'iterations': int(args.num_epochs),
            'learning_rate': float(args.learning_rate),
            'fine_stage_start': 100,
            'signed_distance_backend': 'nearest_vertex',
            'loss_history': 'not_instrumented_to_preserve_original_numeric_path',
            'p1_changes': original_mmm_diagnostics(data_p1, refined_p1),
            'p2_changes': original_mmm_diagnostics(data_p2, refined_p2),
        }
        synchronize_device(args.device)
        sample_wall_seconds = time.perf_counter() - optimization_start
        atomic_npz(out_p1, output_payload(refined_p1, original_genders[0]))
        atomic_npz(out_p2, output_payload(refined_p2, original_genders[1]))
        atomic_json(sidecar, {
            'schema': 'hhc_paper_contact_optimization_sample_v1',
            **completed_identity,
            'output_p1_sha256': sha256_file(out_p1),
            'output_p2_sha256': sha256_file(out_p2),
            'optimization_wall_seconds': sample_wall_seconds,
            'optimization_timing_scope': (
                'CUDA-synchronized optimize_full_body call only; excludes '
                'input loading and output serialization'
            ),
            'diagnostics': diagnostics,
        })
        completed.append(sample_key)
        optimization_wall_seconds.append(sample_wall_seconds)

    runtime_values = np.asarray(optimization_wall_seconds, dtype=np.float64)
    runtime_summary = {
        'samples': len(runtime_values),
        'optimizer_wall_seconds_total': float(runtime_values.sum()),
        'optimizer_wall_seconds_mean_per_sample': float(runtime_values.mean()),
        'optimizer_wall_seconds_median_per_sample': float(np.median(runtime_values)),
        'optimizer_wall_seconds_p95_per_sample': float(
            np.quantile(runtime_values, 0.95)
        ),
        'current_invocation_wall_seconds': time.perf_counter() - invocation_start,
        'scope': (
            'per-sample times are CUDA-synchronized optimize_full_body calls; '
            'the current invocation includes validation, I/O, and resumed skips'
        ),
    }
    atomic_json(output_dir / '_completion_manifest.json', {
        **run_contract,
        'completed_samples': len(completed),
        'completed_sample_keys': completed,
        'complete': len(completed) == len(candidates),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'runtime': runtime_summary,
    })
    print({
        'output_dir': str(output_dir),
        'completed': len(completed),
        'requested': len(candidates),
        'optimizer_wall_seconds_mean_per_sample': runtime_summary[
            'optimizer_wall_seconds_mean_per_sample'
        ],
    })


if __name__ == '__main__':
    main()
