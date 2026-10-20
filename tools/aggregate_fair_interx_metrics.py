#!/usr/bin/env python
"""Aggregate frozen-protocol replication JSON files with official 95% CIs."""

import argparse
import glob
import hashlib
import json
import math
import os
import re

import numpy as np


METRICS = (
    'fid', 'diversity_gt', 'diversity',
    'top1', 'top2', 'top3', 'matching', 'multimodality',
)

REPLICATION_PATTERN = re.compile(r'replication_(\d{2})\.json$')
MODEL_IDENTITY_KEYS = (
    'checkpoint_sha256', 'vq_checkpoint_sha256',
    'normalization_mean_sha256', 'normalization_std_sha256',
    'contact_map_sha256', 'contact_map_contract',
    'seed_strategy', 'sampling_steps', 'temperature', 'cfg_scale',
    'sampler', 'sampler_semantics', 'cascade_order', 'sampling_steps_unit',
    'channel_updates_per_outer_step',
    'model_forward_calls_per_outer_step_with_cfg',
    # Clarified paper sampler identity.  These are upper bounds because a
    # channel that has already resolved skips its forward pass while the
    # outer MaskGIT schedule still completes all requested iterations.
    'maximum_active_channel_stages_per_outer_iteration',
    'maximum_model_forward_calls_per_outer_iteration_with_cfg',
    'resolved_channels_skip_forward_but_not_outer_iterations',
    'contact_guidance', 'contact_guidance_scale', 'generation_length_policy',
    'contact_timing_source',
    'time_steps', 'cond_scale', 'topk_filter_thres', 'length_policy', 'adapter',
    'skeleton_root', 'skeleton_files_sha256', 'method', 'architecture',
    'batch_size', 'strategy', 'cfg_weight', 'projection', 'sampling_process',
    'ddim_eta', 'training_diffusion_steps', 'model_config_sha256',
    'normalization_prefix', 'exporter_sha256', 'model_source_sha256',
    'diffusion_source_sha256', 'gaussian_diffusion_source_sha256',
    'normalizer_source_sha256', 'interaction_source_sha256',
    'adapter_source_sha256', 'request_limit',
    'first_frame_conditioning',
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contact_timing_provenance(metadata, request_path, request_hash):
    """Validate per-rep timing artifacts without treating them as one identity."""
    source = metadata.get('contact_timing_source')
    timing_path = metadata.get('contact_timing_file')
    timing_hash = metadata.get('contact_timing_file_sha256')
    if source is None:
        return
    if source in ('none', 'full_valid_weak'):
        if timing_path not in (None, '') or timing_hash not in (None, ''):
            raise ValueError(
                f'{source} contact timing must not reference an external file'
            )
        return
    if source == 'request_oracle':
        if (not timing_path or
                os.path.abspath(timing_path) != os.path.abspath(request_path)):
            raise ValueError(
                'request_oracle timing must be the same frozen request NPZ'
            )
        if timing_hash != request_hash:
            raise ValueError('request_oracle timing hash differs from request hash')
        return
    if source == 'external_text_predictor':
        if not timing_path or not os.path.isfile(timing_path):
            raise ValueError('external text timing artifact is missing')
        if timing_hash != sha256_file(timing_path):
            raise ValueError('external text timing artifact hash mismatch')
        return
    raise ValueError(f'unknown contact timing source {source!r}')


def validate_replication_provenance(paths, records, allow_legacy=False):
    """Bind every metric row to a unique request and one sampler identity."""
    expected_ids = list(range(len(paths)))
    replication_ids = []
    request_hashes = []
    sampler_identities = []
    legacy_reasons = []

    for path, record in zip(paths, records):
        match = REPLICATION_PATTERN.search(os.path.basename(path))
        if match is None:
            raise ValueError(f'{path}: filename must end in replication_NN.json')
        replication = int(match.group(1))
        replication_ids.append(replication)

        request_path = record.get('requests')
        if not request_path or not os.path.isfile(request_path):
            raise ValueError(f'{path}: frozen request file is missing: {request_path!r}')
        expected_request_name = f'replication_{replication:02d}.npz'
        if os.path.basename(request_path) != expected_request_name:
            raise ValueError(
                f'{path}: expected request {expected_request_name}, got {request_path}'
            )
        actual_request_hash = sha256_file(request_path)
        recorded_request_hash = record.get('requests_sha256')
        if recorded_request_hash is None:
            legacy_reasons.append(f'replication {replication}: missing request hash')
        elif recorded_request_hash != actual_request_hash:
            raise ValueError(
                f'{path}: request hash {recorded_request_hash} != {actual_request_hash}'
            )
        if record.get('generation_request_hash_verified') is not True:
            legacy_reasons.append(
                f'replication {replication}: generation did not embed request hash'
            )
        request_hashes.append(actual_request_hash)

        for generation_key in ('main_generations', 'mm_generations'):
            generation_path = record.get(generation_key)
            recorded_generation_hash = record.get(f'{generation_key}_sha256')
            if not generation_path or not os.path.isfile(generation_path):
                raise ValueError(
                    f'{path}: generated HDF5 is missing ({generation_key}): '
                    f'{generation_path!r}'
                )
            if recorded_generation_hash is None:
                legacy_reasons.append(
                    f'replication {replication}: missing {generation_key} hash'
                )
            else:
                actual_generation_hash = sha256_file(generation_path)
                if recorded_generation_hash != actual_generation_hash:
                    raise ValueError(
                        f'{path}: {generation_key} hash {recorded_generation_hash} '
                        f'!= {actual_generation_hash}'
                    )

        metadata = record.get('main_generation_metadata')
        if not isinstance(metadata, dict):
            legacy_reasons.append(f'replication {replication}: missing generation metadata')
            continue
        if 'checkpoint_sha256' not in metadata:
            legacy_reasons.append(f'replication {replication}: missing checkpoint hash')
        validate_contact_timing_provenance(
            metadata, request_path, actual_request_hash
        )
        sampler_identities.append({
            key: metadata[key] for key in MODEL_IDENTITY_KEYS if key in metadata
        })

    if sorted(replication_ids) != expected_ids:
        raise ValueError(
            f'expected replication IDs {expected_ids}, got {sorted(replication_ids)}'
        )
    if len(set(request_hashes)) != len(request_hashes):
        raise ValueError('replications do not use unique frozen request contents')
    if sampler_identities:
        reference = sampler_identities[0]
        metadata_replications = [
            replication for replication, record in zip(replication_ids, records)
            if isinstance(record.get('main_generation_metadata'), dict)
        ]
        reference_replication = metadata_replications[0]
        for replication, identity in zip(metadata_replications, sampler_identities):
            if identity != reference:
                raise ValueError(
                    f'replication {replication}: model/sampler identity differs '
                    f'from replication {reference_replication}'
                )
    if legacy_reasons and not allow_legacy:
        raise ValueError(
            'provenance is incomplete; regenerate/re-score the artifacts or pass '
            f'--allow-legacy-provenance for an explicitly labelled audit: {legacy_reasons}'
        )
    return {
        'legacy_provenance_allowed': bool(allow_legacy),
        'legacy_provenance_reasons': legacy_reasons,
        'replication_ids': replication_ids,
        'request_sha256': request_hashes,
        'generation_files_sha256_verified': not any(
            'generations hash' in reason for reason in legacy_reasons
        ),
        'sampler_identity': sampler_identities[0] if sampler_identities else None,
        'unique_requests_verified': len(set(request_hashes)) == len(request_hashes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True, help='JSON paths or glob patterns')
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-replications', type=int, default=20)
    parser.add_argument(
        '--expected-samples', type=int, default=1696,
        help=(
            'Main samples per replication. The strict official table uses '
            '1696; pre-registered complete-batch subsets must pass their '
            'frozen request count explicitly.'
        ),
    )
    parser.add_argument(
        '--allow-legacy-provenance', action='store_true',
        help='Permit old metric JSONs without embedded request/model hashes.',
    )
    args = parser.parse_args()

    candidates = {path for pattern in args.inputs for path in glob.glob(pattern)}
    # A method directory can also contain replication_00_physical.json and
    # representation-control audits.  They must never be pooled into the
    # official text-to-motion aggregate.
    paths = sorted(
        path for path in candidates
        if REPLICATION_PATTERN.search(os.path.basename(path))
    )
    if len(paths) != args.expected_replications:
        raise ValueError(
            f'expected {args.expected_replications} replication files, got {len(paths)}'
        )
    records = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as handle:
            record = json.load(handle)
        if record.get('samples') != args.expected_samples:
            raise ValueError(
                f'{path}: expected {args.expected_samples} samples, '
                f'got {record.get("samples")}'
            )
        if record.get('mm_samples') != 100 or record.get('mm_repeats') != 30:
            raise ValueError(f'{path}: expected MM contract 100 x 30')
        missing = [metric for metric in METRICS if metric not in record]
        if missing:
            raise ValueError(f'{path}: missing metrics {missing}')
        records.append(record)

    provenance = validate_replication_provenance(
        paths, records, allow_legacy=args.allow_legacy_provenance
    )

    summary = {
        'replications': len(records),
        'samples_per_replication': args.expected_samples,
        'confidence': 0.95,
        'ci_definition': '1.96 * population_std / sqrt(replications)',
        'inputs': [os.path.abspath(path) for path in paths],
        'metrics': {},
        'provenance': provenance,
    }
    for metric in METRICS:
        values = np.asarray([record[metric] for record in records], dtype=np.float64)
        population_std = float(np.std(values))
        summary['metrics'][metric] = {
            'mean': float(np.mean(values)),
            'ci95': float(1.96 * population_std / math.sqrt(len(values))),
            'population_std': population_std,
            'values': values.tolist(),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
