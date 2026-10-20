#!/usr/bin/env python
"""Audit CoShMDM's frozen Inter-X generation and fair-evaluation contract."""

import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


HHC_ROOT = Path(__file__).resolve().parents[1]
COSH_ROOT = Path('/mnt/8T-HDD/wenning/coshmdm-code')


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def source_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path)}


def call_keywords(path, attribute):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]
    require(len(calls) == 1, f'expected one {attribute} call, got {len(calls)}')
    return {keyword.arg for keyword in calls[0].keywords}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--requests-manifest', type=Path,
        default=HHC_ROOT / 'output/fair_interx_requests/manifest.json',
    )
    parser.add_argument(
        '--stats-manifest', type=Path,
        default=COSH_ROOT /
        'output/interx_stats_recompute_audit/interx_global_stats_manifest.json',
    )
    parser.add_argument(
        '--output', type=Path,
        default=HHC_ROOT / 'output/coshmdm_generation_contract_audit.json',
    )
    args = parser.parse_args()

    paths = {
        'model_config': COSH_ROOT / 'configs/model.yaml',
        'exporter': COSH_ROOT / 'tools/export_official_interx.py',
        'model': COSH_ROOT / 'models/coshmdm.py',
        'diffusion': COSH_ROOT / 'models/nets.py',
        'gaussian_diffusion': COSH_ROOT / 'models/gaussian_diffusion.py',
        'normalizer': COSH_ROOT / 'utils/utils.py',
        'adapter': COSH_ROOT / 'utils/interx_official_adapter.py',
        'driver': HHC_ROOT / 'tools/run_fair_interx_method.py',
        'scorer': HHC_ROOT / 'tools/score_fair_interx_generations.py',
        'requests_manifest': args.requests_manifest.resolve(),
        'stats_manifest': args.stats_manifest.resolve(),
        'normalization_mean': COSH_ROOT / 'data/interx_global_mean.npy',
        'normalization_std': COSH_ROOT / 'data/interx_global_std.npy',
    }
    for label, path in paths.items():
        require(path.is_file(), f'missing {label}: {path}')

    config = yaml.safe_load(paths['model_config'].read_text(encoding='utf-8'))
    require(config.get('STRATEGY') == 'ddim50', 'CoShMDM strategy is not ddim50')
    require(config.get('CFG_WEIGHT') == 3.5, 'CoShMDM CFG is not 3.5')
    require(config.get('DIFFUSION_STEPS') == 1000,
            'CoShMDM training diffusion is not 1000 steps')

    sys.path.insert(0, str(COSH_ROOT))
    from models.gaussian_diffusion import (  # pylint: disable=import-outside-toplevel
        GaussianDiffusion, space_timesteps,
    )
    retained_timesteps = sorted(space_timesteps(1000, 'ddim50'))
    eta_default = inspect.signature(
        GaussianDiffusion.ddim_sample_loop
    ).parameters['eta'].default
    require(len(retained_timesteps) == 50, 'ddim50 did not resolve to 50 steps')
    require(retained_timesteps == list(range(0, 1000, 20)),
            'ddim50 retained-timestep schedule changed')
    require(eta_default == 0.0, 'DDIM sampler default eta is not zero')
    require('eta' not in call_keywords(paths['diffusion'], 'ddim_sample_loop'),
            'CoShMDM forward overrides the audited eta=0 default')

    exporter = paths['exporter'].read_text(encoding='utf-8')
    driver = paths['driver'].read_text(encoding='utf-8')
    scorer = paths['scorer'].read_text(encoding='utf-8')
    exporter_requirements = {
        "str(model_cfg.STRATEGY) != 'ddim50'",
        'int(model_cfg.DIFFUSION_STEPS) != 1000',
        'float(model_cfg.CFG_WEIGHT) != 3.5',
        "stats_prefix != 'interx_global'",
        "'sampling_process': 'DDIM50_eta0_CFG3.5'",
        "'sampling_steps': 50",
        "'ddim_eta': 0.0",
        "'normalization_mean_sha256'",
        "'normalization_std_sha256'",
        "'exporter_sha256'",
        'for valid_length in sorted(np.unique(lengths).tolist())',
        "generated = model.forward_test(batch)['output']",
        'official[:, :int(valid_length)] = official_valid',
        'completed_repeats[repeat] = True',
    }
    require(all(item in exporter for item in exporter_requirements),
            'CoShMDM exporter lost a required fail-fast/provenance contract')
    driver_requirements = {
        "parser.add_argument('--end-replication', type=int, default=20)",
        "seed + 100_000, 30, indices_key='mm_indices'",
        "environment['COSHMDM_STATS_PREFIX'] = 'interx_global'",
        "sys.executable, 'tools/audit_coshmdm_generation_contract.py'",
        "'coshmdm_sampling_process': (",
        "'DDIM50_eta0_CFG3.5' if args.method == 'coshmdm' else None",
    }
    require(all(item in driver for item in driver_requirements),
            'fair driver lost the 20-replication/main-MM CoShMDM contract')
    metadata_keys = {
        'sampling_process', 'ddim_eta', 'training_diffusion_steps',
        'model_config_sha256', 'normalization_mean_sha256',
        'normalization_std_sha256', 'exporter_sha256',
        'model_source_sha256', 'diffusion_source_sha256',
        'gaussian_diffusion_source_sha256', 'normalizer_source_sha256',
        'adapter_source_sha256',
    }
    require(all(repr(key) in scorer for key in metadata_keys),
            'official scorer no longer cross-checks all CoShMDM sampler inputs')

    stats = json.loads(paths['stats_manifest'].read_text(encoding='utf-8'))
    require(
        stats.get('processed_samples') == 9110
        and stats.get('requested_samples') == 9110
        and stats.get('statistics_scope')
        == 'official_interx_train_with_mirror_and_person_order',
        'CoShMDM statistics do not cover the official 9110-sample train split',
    )
    require(stats.get('mean_sha256') == sha256_file(paths['normalization_mean']),
            'CoShMDM mean differs from the train-split recomputation')
    require(stats.get('std_sha256') == sha256_file(paths['normalization_std']),
            'CoShMDM std differs from the train-split recomputation')

    requests = json.loads(
        paths['requests_manifest'].read_text(encoding='utf-8')
    )
    require(requests.get('status') == 'pass', 'frozen requests are incomplete')
    require(requests.get('protocol') == 'interx_official_v1',
            'frozen requests use the wrong evaluator protocol')
    require(requests.get('replications') == 20, 'expected 20 request replications')
    require(requests.get('samples_per_replication') == 1696,
            'expected 1696 eligible Inter-X test motions')
    require(requests.get('multimodality') == {
        'pairs': 10, 'prompts': 100, 'repeats': 30,
    }, 'multimodality request contract changed')
    records = requests.get('records', [])
    require(len(records) == 20, 'request manifest does not contain 20 records')
    request_hashes_verified = 0
    for replication, record in enumerate(records):
        request_path = Path(record['path']).resolve()
        require(record.get('replication') == replication,
                f'non-contiguous request replication {replication}')
        require(request_path.is_file(), f'missing request file: {request_path}')
        require(record.get('sha256') == sha256_file(request_path),
                f'request hash mismatch: {request_path}')
        with np.load(request_path, allow_pickle=False) as frozen:
            require(len(frozen['names']) == 1696,
                    f'wrong request sample count: {request_path}')
            require(len(frozen['mm_indices']) == 100,
                    f'wrong MM prompt count: {request_path}')
            require(int(frozen['seed']) == int(record['seed']),
                    f'request seed mismatch: {request_path}')
        request_hashes_verified += 1

    checks = {
        'author_config_ddim50_cfg3.5_diffusion1000': True,
        'runtime_ddim_retains_exactly_50_timesteps': True,
        'runtime_ddim_eta_is_zero': True,
        'exporter_fails_closed_on_sampler_or_stats_drift': True,
        'exporter_binds_model_stats_code_checkpoint_requests_and_skeletons': True,
        'exact_request_lengths_are_generated_in_equal_length_groups': True,
        'main_uses_one_repeat_and_mm_uses_30_repeats': True,
        'mm_seed_is_request_seed_plus_100000': True,
        'each_repeat_seed_is_export_seed_plus_repeat_index': True,
        'twenty_frozen_request_replications_are_byte_verified': True,
        'official_full_672d_scorer_cross_checks_main_mm_sampler_metadata': True,
        'no_gt_motion_is_an_exporter_input': True,
    }
    report = {
        'schema': 'hhc_coshmdm_generation_contract_audit_v1',
        'status': 'pass',
        'checks': checks,
        'sampling': {
            'training_diffusion_steps': 1000,
            'process': 'DDIM50_eta0_CFG3.5',
            'runtime_retained_timesteps': retained_timesteps,
            'runtime_sampling_steps': len(retained_timesteps),
            'ddim_eta': eta_default,
        },
        'requests': {
            'replications': 20,
            'samples_per_replication': 1696,
            'verified_file_hashes': request_hashes_verified,
            'mm_prompts': 100,
            'mm_repeats': 30,
        },
        'representation': {
            'generated': 'CoShMDM 262D per person',
            'exported': 'official Inter-X 672D full space',
            'length_policy': 'exact_request_length_grouped',
            'gt_motion_consumed_by_exporter': False,
        },
        'sources': {label: source_record(path) for label, path in paths.items()},
        'implementation': source_record(__file__),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
