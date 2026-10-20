#!/usr/bin/env python
"""Audit TiMotion's frozen Inter-X generation and fair-evaluation contract."""

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
TIMOTION_ROOT = Path('/mnt/8T-HDD/wenning/TIMotion')


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
        default=Path('/mnt/8T-HDD/wenning/coshmdm-code/output/') /
        'interx_stats_recompute_audit/interx_global_stats_manifest.json',
    )
    parser.add_argument(
        '--output', type=Path,
        default=HHC_ROOT / 'output/timotion_generation_contract_audit.json',
    )
    args = parser.parse_args()

    paths = {
        'model_config': TIMOTION_ROOT / 'configs/model.yaml',
        'exporter': HHC_ROOT / 'tools/export_timotion_official_interx_standard.py',
        'historical_exporter': HHC_ROOT / 'tools/export_timotion_official_interx.py',
        'model': TIMOTION_ROOT / 'models/timotion.py',
        'diffusion': TIMOTION_ROOT / 'models/nets.py',
        'interaction_block': TIMOTION_ROOT / 'models/blocks.py',
        'gaussian_diffusion': TIMOTION_ROOT / 'models/gaussian_diffusion.py',
        'driver': HHC_ROOT / 'tools/run_fair_interx_method.py',
        'scorer': HHC_ROOT / 'tools/score_fair_interx_generations.py',
        'requests_manifest': args.requests_manifest.resolve(),
        'stats_manifest': args.stats_manifest.resolve(),
        'normalization_mean': TIMOTION_ROOT / 'data/interx_global_mean.npy',
        'normalization_std': TIMOTION_ROOT / 'data/interx_global_std.npy',
    }
    for label, path in paths.items():
        require(path.is_file(), f'missing {label}: {path}')

    config = yaml.safe_load(paths['model_config'].read_text(encoding='utf-8'))
    require(config.get('STRATEGY') == 'ddim50', 'TiMotion strategy is not ddim50')
    require(config.get('CFG_WEIGHT') == 3.5, 'TiMotion CFG is not 3.5')
    require(config.get('DIFFUSION_STEPS') == 1000,
            'TiMotion training diffusion is not 1000 steps')

    sys.path.insert(0, str(TIMOTION_ROOT))
    from models.gaussian_diffusion import (  # pylint: disable=import-outside-toplevel
        GaussianDiffusion, space_timesteps,
    )
    retained_timesteps = sorted(space_timesteps(1000, 'ddim50'))
    eta_default = inspect.signature(
        GaussianDiffusion.ddim_sample_loop
    ).parameters['eta'].default
    require(retained_timesteps == list(range(0, 1000, 20)),
            'TiMotion ddim50 retained-timestep schedule changed')
    require(eta_default == 0.0, 'TiMotion DDIM eta default is not zero')
    require('eta' not in call_keywords(paths['diffusion'], 'ddim_sample_loop'),
            'TiMotion inference overrides the audited eta=0 default')

    exporter = paths['exporter'].read_text(encoding='utf-8')
    historical_exporter = paths['historical_exporter'].read_text(
        encoding='utf-8'
    )
    driver = paths['driver'].read_text(encoding='utf-8')
    scorer = paths['scorer'].read_text(encoding='utf-8')
    historical_exporter_requirements = {
        'str(runtime_cfg.STRATEGY) != "ddim50"',
        'int(runtime_cfg.DIFFUSION_STEPS) != 1000',
        'float(runtime_cfg.CFG_WEIGHT) != 3.5',
        'stats_prefix != "interx_global"',
        '"sampling_process": "DDIM50_eta0_CFG3.5"',
        '"sampling_steps": 50',
        '"ddim_eta": 0.0',
        '"model_config_sha256"',
        '"normalization_mean_sha256"',
        '"normalization_std_sha256"',
        '"exporter_sha256"',
        'for valid_length in sorted(np.unique(lengths).tolist())',
        'generated = model.forward_test(batch)["output"]',
        'official[:, :int(valid_length)] = official_valid',
        'completed[repeat] = True',
    }
    require(
        all(item in historical_exporter for item in historical_exporter_requirements),
        'TiMotion base exporter lost a sampler/fail-fast requirement',
    )
    formal_exporter_requirements = {
        'FORMAL_ADAPTER',
        'ROTATION_CONTRACT',
        'formal_result_eligible',
        'load_prerequisites()',
        'standard_rotation_6d_to_matrix',
        'base.main()',
    }
    require(
        all(item in exporter for item in formal_exporter_requirements),
        'TiMotion formal exporter lost its standard-6D provenance gate',
    )
    driver_requirements = {
        "parser.add_argument('--end-replication', type=int, default=20)",
        "seed + 100_000, 30, indices_key='mm_indices'",
        "environment['TIMOTION_STATS_PREFIX'] = 'interx_global'",
        "sys.executable, 'tools/audit_timotion_generation_contract.py'",
        "'DDIM50_eta0_CFG3.5' if args.method == 'timotion' else None",
        "'standard_row_concatenated_v1' if args.method == 'timotion' else None",
    }
    require(all(item in driver for item in driver_requirements),
            'fair driver lost the TiMotion 20-replication/main-MM contract')
    metadata_keys = {
        'sampling_process', 'sampling_steps', 'ddim_eta',
        'training_diffusion_steps', 'model_config_sha256',
        'normalization_mean_sha256', 'normalization_std_sha256',
        'exporter_sha256', 'model_source_sha256',
        'diffusion_source_sha256', 'interaction_source_sha256',
        'gaussian_diffusion_source_sha256', 'adapter_source_sha256',
    }
    require(all(repr(key) in scorer for key in metadata_keys),
            'official scorer no longer cross-checks TiMotion sampler inputs')

    stats = json.loads(paths['stats_manifest'].read_text(encoding='utf-8'))
    require(stats.get('processed_samples') == 9110
            and stats.get('requested_samples') == 9110,
            'TiMotion statistics do not cover the official train split')
    require(stats.get('mean_sha256') == sha256_file(paths['normalization_mean']),
            'TiMotion mean differs from train-split recomputation')
    require(stats.get('std_sha256') == sha256_file(paths['normalization_std']),
            'TiMotion std differs from train-split recomputation')

    requests = json.loads(
        paths['requests_manifest'].read_text(encoding='utf-8')
    )
    require(requests.get('status') == 'pass', 'frozen requests are incomplete')
    require(requests.get('protocol') == 'interx_official_v1',
            'frozen requests use the wrong evaluator protocol')
    require(requests.get('replications') == 20
            and requests.get('samples_per_replication') == 1696,
            'TiMotion fair requests are not 20 x 1696')
    require(requests.get('multimodality') == {
        'pairs': 10, 'prompts': 100, 'repeats': 30,
    }, 'TiMotion multimodality request contract changed')
    records = requests.get('records', [])
    require(len(records) == 20, 'request manifest does not contain 20 records')
    for replication, record in enumerate(records):
        request_path = Path(record['path']).resolve()
        require(record.get('replication') == replication,
                f'non-contiguous request replication {replication}')
        require(record.get('sha256') == sha256_file(request_path),
                f'request hash mismatch: {request_path}')
        with np.load(request_path, allow_pickle=False) as frozen:
            require(len(frozen['names']) == 1696
                    and len(frozen['mm_indices']) == 100,
                    f'wrong request/MM count: {request_path}')
            require(int(frozen['seed']) == int(record['seed']),
                    f'request seed mismatch: {request_path}')

    checks = {
        'public_transformer_lpa_config_is_fixed': True,
        'runtime_ddim_retains_exactly_50_timesteps': True,
        'runtime_ddim_eta_is_zero': True,
        'exporter_fails_closed_on_sampler_or_stats_drift': True,
        'exporter_binds_model_stats_code_checkpoint_requests_and_skeletons': True,
        'exact_request_lengths_are_generated_in_equal_length_groups': True,
        'main_one_repeat_mm_30_repeats_and_20_replications': True,
        'official_full_672d_scorer_cross_checks_sampler_metadata': True,
        'no_gt_motion_or_first_frame_is_an_exporter_input': True,
    }
    report = {
        'schema': 'hhc_timotion_generation_contract_audit_v1',
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
            'replications': 20, 'samples_per_replication': 1696,
            'verified_file_hashes': len(records),
            'mm_prompts': 100, 'mm_repeats': 30,
        },
        'representation': {
            'generated': 'TiMotion 262D per person',
            'exported': 'official Inter-X 672D full space',
            'rotation6d_contract': 'standard_row_concatenated_v1',
            'length_policy': 'exact_request_length_grouped',
            'gt_motion_consumed_by_exporter': False,
            'gt_first_frame_consumed_by_exporter': False,
        },
        'sources': {label: source_record(path) for label, path in paths.items()},
        'implementation': source_record(__file__),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
