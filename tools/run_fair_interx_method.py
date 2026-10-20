#!/usr/bin/env python
"""Run one method through frozen Inter-X main/MM generation and common scoring."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np


HHC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERMASK_ROOT = '/mnt/8T-HDD/wenning/InterMask'
COSHMDM_ROOT = '/mnt/8T-HDD/wenning/coshmdm-code'
INTERACT2AR_ROOT = '/mnt/8T-HDD/wenning/Interact2Ar'
TIMOTION_ROOT = '/mnt/8T-HDD/wenning/TIMotion'


def git_repository_record(root):
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=root, text=True
    ).splitlines()
    tracked_diff = subprocess.check_output(
        ['git', 'diff', '--binary', 'HEAD'], cwd=root
    )
    return {
        'path': root,
        'revision': revision,
        'dirty': bool(status),
        'status': status,
        'tracked_diff_sha256': hashlib.sha256(tracked_diff).hexdigest(),
    }


def artifact_record(path):
    absolute = os.path.abspath(path)
    digest = hashlib.sha256()
    with open(absolute, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {
        'path': absolute,
        'bytes': os.path.getsize(absolute),
        'sha256': digest.hexdigest(),
    }


def freeze_implementation_records(implementation, method_output):
    snapshot_dir = os.path.join(method_output, '_provenance_sources')
    os.makedirs(snapshot_dir, exist_ok=True)
    output = {}
    for name, source_path in implementation.items():
        source_record = artifact_record(source_path)
        destination = os.path.join(
            snapshot_dir,
            f"{source_record['sha256']}__{os.path.basename(source_path)}",
        )
        if os.path.isfile(destination):
            if artifact_record(destination)['sha256'] != source_record['sha256']:
                raise ValueError(f'corrupt implementation snapshot: {destination}')
        else:
            temporary = destination + f'.tmp.{os.getpid()}'
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, destination)
        output[name] = {
            **artifact_record(destination),
            'source_path_at_execution': os.path.abspath(source_path),
            'content_addressed_snapshot': True,
        }
    return output


def validate_launch_sources_unchanged(provenance_record):
    mismatches = {}
    for name, record in provenance_record['implementation'].items():
        source = record.get('source_path_at_execution')
        if not source or not os.path.isfile(source):
            mismatches[name] = 'source_missing'
            continue
        current_hash = artifact_record(source)['sha256']
        if current_hash != record['sha256']:
            mismatches[name] = {
                'launch_sha256': record['sha256'],
                'completion_sha256': current_hash,
            }
    if mismatches:
        raise RuntimeError(
            'implementation source changed during the fair run: '
            + json.dumps(mismatches, sort_keys=True)
        )


def provenance(args, method_output):
    evaluator = os.path.join(
        HHC_ROOT, 'checkpoints/hhi/text_mot_match/model/finest.tar'
    )
    protocol = os.path.join(HHC_ROOT, 'configs/fair_interx_protocol.yaml')
    if args.method == 'intermask':
        artifacts = {
            'transformer': os.path.join(
                INTERMASK_ROOT,
                'checkpoints/interx/trans_default/model/best_fid.tar',
            ),
            'vq': os.path.join(
                INTERMASK_ROOT, 'checkpoints/interx/vq_default/model/best_fid.tar'
            ),
        }
    elif args.method == 'hhc':
        artifacts = {
            'transformer': args.checkpoint,
            'vq': args.vq_checkpoint,
            'normalization_mean': os.path.join(
                os.path.abspath(args.hhc_stats_dir),
                f'{args.hhc_stats_prefix}_Mean.npy',
            ),
            'normalization_std': os.path.join(
                os.path.abspath(args.hhc_stats_dir),
                f'{args.hhc_stats_prefix}_Std.npy',
            ),
            'vq_training_manifest': os.path.join(
                os.path.dirname(os.path.abspath(args.vq_checkpoint)),
                'training_manifest.json',
            ),
        }
        if args.hhc_mgt_contract == 'current':
            artifacts['mgt_training_manifest'] = os.path.join(
                os.path.dirname(os.path.abspath(args.checkpoint)),
                'training_manifest.json',
            )
        if args.selection_manifest:
            artifacts['mgt_selection_manifest'] = args.selection_manifest
        if os.path.dirname(os.path.abspath(args.vq_checkpoint)) == os.path.join(
                HHC_ROOT, 'checkpoints/vq_paper'):
            artifacts['vq_launch_source_snapshot'] = os.path.join(
                HHC_ROOT, 'output/vq-interx-sep/'
                '2026-08-26-00-09-40_'
                'paper_vq_8096_150k_seed123_exact_rawstats_ema_safe/'
                'source/complete_source_snapshot_manifest.json'
            )
        if args.contact_guidance:
            artifacts['contact_map'] = args.hhc_contact_map
            if args.hhc_contact_map_contract == 'paper_formula_train_only':
                artifacts['contact_map_manifest'] = os.path.join(
                    os.path.dirname(os.path.abspath(args.hhc_contact_map)),
                    'codebook', '_cache_manifest.json',
                )
            else:
                artifacts['contact_map_manifest'] = os.path.join(
                    HHC_ROOT, 'output',
                    'hhc_contact_guidance_map_audit.json',
                )
            if args.hhc_contact_timing_source == 'external_text_predictor':
                artifacts['contact_timing_file'] = args.hhc_contact_timing_file
    elif args.method == 'coshmdm':
        artifacts = {
            'model': args.checkpoint,
            'normalization_mean': os.path.join(
                COSHMDM_ROOT, 'data/interx_global_mean.npy'
            ),
            'normalization_std': os.path.join(
                COSHMDM_ROOT, 'data/interx_global_std.npy'
            ),
            'training_split': os.path.join(
                COSHMDM_ROOT, 'data/interx_processed/train.txt'
            ),
            'normalization_recompute_manifest': os.path.join(
                COSHMDM_ROOT,
                'output/interx_stats_recompute_audit/interx_global_stats_manifest.json',
            ),
            'training_manifest': os.path.join(
                COSHMDM_ROOT,
                'checkpoints/CoShMDM-InterX-fair-statsfix-seed123-bs32x2/meta/training_manifest.json',
            ),
            'model_config': os.path.join(COSHMDM_ROOT, 'configs/model.yaml'),
            'generation_contract_audit': os.path.join(
                HHC_ROOT, 'output/coshmdm_generation_contract_audit.json'
            ),
            'milestone_preservation_manifest': os.path.join(
                HHC_ROOT, 'output/coshmdm_milestone_preservation.json'
            ),
        }
        capped_selection = os.path.join(
            HHC_ROOT, 'output/coshmdm_interx_val_selection.json'
        )
        capped_budget = os.path.join(
            HHC_ROOT, 'output/coshmdm_training_budget_cap.json'
        )
        if os.path.isfile(capped_selection):
            artifacts['validation_selection_manifest'] = capped_selection
        if os.path.isfile(capped_budget):
            artifacts['user_training_budget_cap'] = capped_budget
    elif args.method == 'timotion':
        checkpoint_root = os.path.dirname(
            os.path.dirname(os.path.abspath(args.checkpoint))
        )
        artifacts = {
            'model': args.checkpoint,
            'normalization_mean': os.path.join(
                TIMOTION_ROOT, 'data/interx_global_mean.npy'
            ),
            'normalization_std': os.path.join(
                TIMOTION_ROOT, 'data/interx_global_std.npy'
            ),
            'training_split': os.path.join(TIMOTION_ROOT, 'data/InterX/train.txt'),
            'normalization_recompute_manifest': os.path.join(
                COSHMDM_ROOT,
                'output/interx_stats_recompute_audit/interx_global_stats_manifest.json',
            ),
            'training_manifest': os.path.join(
                checkpoint_root, 'meta/training_manifest.json'
            ),
            'model_config': os.path.join(TIMOTION_ROOT, 'configs/model.yaml'),
            'generation_contract_audit': os.path.join(
                HHC_ROOT, 'output/timotion_generation_contract_audit.json'
            ),
        }
        if args.selection_manifest:
            artifacts['validation_selection_manifest'] = args.selection_manifest
    else:
        artifacts = {
            'model': args.checkpoint,
            'author_checkpoint_manifest': os.path.join(
                INTERACT2AR_ROOT, 'assets/checkpoints.json'
            ),
            'official_test_split': os.path.join(
                INTERACT2AR_ROOT, 'data/splits/test.txt'
            ),
            'locked_dependencies': os.path.join(INTERACT2AR_ROOT, 'uv.lock'),
        }
    artifacts.update({'evaluator': evaluator, 'protocol': protocol})
    artifacts['requests_manifest'] = (
        args.requests_manifest
        if args.requests_manifest
        else os.path.join(os.path.abspath(args.requests_dir), 'manifest.json')
    )
    implementation = {
        'driver': os.path.join(HHC_ROOT, 'tools/run_fair_interx_method.py'),
        'semantic_scorer': os.path.join(
            HHC_ROOT, 'tools/score_fair_interx_generations.py'
        ),
        'physical_scorer': os.path.join(
            HHC_ROOT, 'tools/score_fair_interx_physical.py'
        ),
        'aggregator': os.path.join(
            HHC_ROOT, 'tools/aggregate_fair_interx_metrics.py'
        ),
        'generation_timing_aggregator': os.path.join(
            HHC_ROOT, 'tools/aggregate_fair_interx_generation_timings.py'
        ),
        'multimodality_analyzer': os.path.join(
            HHC_ROOT, 'tools/analyze_fair_interx_multimodality.py'
        ),
        'multimodality_aggregator': os.path.join(
            HHC_ROOT, 'tools/aggregate_fair_interx_multimodality_audits.py'
        ),
    }
    if args.method == 'hhc' and args.selection_manifest:
        with open(args.selection_manifest, encoding='utf-8') as handle:
            hhc_selection = json.load(handle)
        if hhc_selection.get('schema') == 'hhc_official_interx_validation_selection_v1':
            selector_name = 'select_hhc_checkpoint_on_official_interx_val.py'
        else:
            selector_name = 'select_best_mgt_checkpoint.py'
        implementation['checkpoint_selector'] = os.path.join(
            HHC_ROOT, 'tools', selector_name
        )
    elif args.method == 'timotion' and args.selection_manifest:
        implementation['checkpoint_selector'] = os.path.join(
            HHC_ROOT, 'tools/select_timotion_checkpoint_on_interx_val.py'
        )
    if args.method == 'intermask':
        implementation['exporter'] = os.path.join(
            INTERMASK_ROOT, 'tools_export_official_interx.py'
        )
    elif args.method == 'hhc':
        implementation['exporter'] = os.path.join(
            HHC_ROOT, 'tools/export_hhc_official_interx.py'
        )
    elif args.method == 'coshmdm':
        implementation.update({
            'exporter': os.path.join(
                COSHMDM_ROOT, 'tools/export_official_interx.py'
            ),
            'official_adapter': os.path.join(
                COSHMDM_ROOT, 'utils/interx_official_adapter.py'
            ),
            'representation_projector': os.path.join(
                COSHMDM_ROOT, 'tools/project_official_interx_h5.py'
            ),
            'training_data_converter': os.path.join(
                HHC_ROOT, 'tools/prepare_coshmdm_interx.py'
            ),
            'normalization_generator': os.path.join(
                COSHMDM_ROOT, 'tools/compute_interx_stats.py'
            ),
            'official_model': os.path.join(COSHMDM_ROOT, 'models/coshmdm.py'),
            'official_diffusion': os.path.join(COSHMDM_ROOT, 'models/nets.py'),
            'gaussian_diffusion': os.path.join(
                COSHMDM_ROOT, 'models/gaussian_diffusion.py'
            ),
            'normalizer': os.path.join(COSHMDM_ROOT, 'utils/utils.py'),
            'generation_contract_auditor': os.path.join(
                HHC_ROOT, 'tools/audit_coshmdm_generation_contract.py'
            ),
            'milestone_preserver': os.path.join(
                HHC_ROOT, 'tools/preserve_coshmdm_milestones.py'
            ),
        })
    elif args.method == 'timotion':
        implementation.update({
            'exporter': os.path.join(
                HHC_ROOT, 'tools/export_timotion_official_interx_standard.py'
            ),
            'historical_exporter': os.path.join(
                HHC_ROOT, 'tools/export_timotion_official_interx.py'
            ),
            'official_model': os.path.join(TIMOTION_ROOT, 'models/timotion.py'),
            'official_denoiser': os.path.join(TIMOTION_ROOT, 'models/nets.py'),
            'official_interaction_block': os.path.join(
                TIMOTION_ROOT, 'models/blocks.py'
            ),
            'gaussian_diffusion': os.path.join(
                TIMOTION_ROOT, 'models/gaussian_diffusion.py'
            ),
            'generation_contract_auditor': os.path.join(
                HHC_ROOT, 'tools/audit_timotion_generation_contract.py'
            ),
            'fair_training_runner': os.path.join(
                TIMOTION_ROOT, 'tools/train_interx_fair.py'
            ),
            'training_data_converter': os.path.join(
                HHC_ROOT, 'tools/prepare_coshmdm_interx.py'
            ),
        })
    else:
        implementation.update({
            'exporter': os.path.join(
                HHC_ROOT, 'tools/export_interact2ar_official_interx.py'
            ),
            'official_model': os.path.join(
                INTERACT2AR_ROOT, 'src/models/interact2ar.py'
            ),
            'official_lightning_wrapper': os.path.join(
                INTERACT2AR_ROOT, 'src/models/wrapper.py'
            ),
            'official_environment': os.path.join(
                INTERACT2AR_ROOT, 'pyproject.toml'
            ),
        })
    artifact_records = {
        name: artifact_record(path) for name, path in artifacts.items()
    }
    if args.method in ('hhc', 'timotion') and args.selection_manifest:
        with open(args.selection_manifest, 'r', encoding='utf-8') as handle:
            selection = json.load(handle)
        selected_path = selection.get('selected_checkpoint')
        if (not selected_path or
                os.path.abspath(selected_path) != os.path.abspath(args.checkpoint)):
            raise ValueError(
                f'{args.method} selection manifest does not select --checkpoint: '
                f'{selected_path!r} vs {args.checkpoint!r}'
            )
        checkpoint_artifact_key = (
            'transformer' if args.method == 'hhc' else 'model'
        )
        if (selection.get('selected_checkpoint_sha256') !=
                artifact_records[checkpoint_artifact_key]['sha256']):
            raise ValueError(
                f'{args.method} selection-manifest checkpoint hash mismatch'
            )

    return {
        'repositories': {
            'hhc': git_repository_record(HHC_ROOT),
            'intermask': git_repository_record(INTERMASK_ROOT),
            'coshmdm': git_repository_record(COSHMDM_ROOT),
            'interact2ar': git_repository_record(INTERACT2AR_ROOT),
            'timotion': git_repository_record(TIMOTION_ROOT),
        },
        'artifacts': artifact_records,
        'implementation': freeze_implementation_records(
            implementation, method_output,
        ),
        'source_validation_contract': (
            'fair_interx_content_addressed_implementation_snapshots_v1'
        ),
        'repository_records_role': (
            'historical context and training-generation equality; immutable '
            'implementation bytes are validated through snapshots'
        ),
    }


def run(command, cwd, env, dry_run):
    record = {
        'cwd': cwd, 'command': command,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(record), flush=True)
    if dry_run:
        return None
    started = time.perf_counter()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    record['elapsed_seconds'] = time.perf_counter() - started
    record['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    return record


def atomic_json(path, payload):
    temporary = path + f'.tmp.{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def report_is_current(path, expected_paths, dependencies, expected_values=None):
    """Return true only when a report belongs to, and postdates, its inputs."""
    if not os.path.isfile(path) or any(not os.path.isfile(item) for item in dependencies):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            report = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    for key, expected in expected_paths.items():
        actual = report.get(key)
        if actual is None or os.path.abspath(actual) != os.path.abspath(expected):
            return False
        recorded_hash = report.get(f'{key}_sha256')
        if recorded_hash is None or recorded_hash != artifact_record(expected)['sha256']:
            return False
    for key, expected in (expected_values or {}).items():
        if report.get(key) != expected:
            return False
    report_mtime = os.stat(path).st_mtime_ns
    return report_mtime >= max(os.stat(item).st_mtime_ns for item in dependencies)


def exporter_command(args, request, output, seed, repeats, indices_key=None):
    common = [
        '--requests', request, '--output', output, '--batch-size', str(args.batch_size),
        '--repeats', str(repeats), '--seed', str(seed), '--resume',
    ]
    if indices_key:
        common += ['--indices-key', indices_key]
    if args.method == 'hhc':
        command = [
            sys.executable, 'tools/export_hhc_official_interx.py',
            '--checkpoint', os.path.abspath(args.checkpoint),
            '--vq-checkpoint', os.path.abspath(args.vq_checkpoint),
            '--stats-dir', os.path.abspath(args.hhc_stats_dir),
            '--stats-prefix', args.hhc_stats_prefix,
            '--vq-contract', args.hhc_vq_contract,
            '--mgt-contract', args.hhc_mgt_contract,
            '--mgt-shape' if args.hhc_mgt_shape else '--no-mgt-shape',
            '--inference-conditioning-contract',
            args.hhc_inference_conditioning_contract,
            '--contact-map', os.path.abspath(args.hhc_contact_map),
            '--contact-map-contract', args.hhc_contact_map_contract,
            '--device', f'cuda:{args.gpu}',
        ]
        command += ['--sampling-steps', str(args.sampling_steps)]
        command += [
            '--temperature', str(args.temperature),
            '--cfg-scale', str(args.cfg_scale),
            '--contact-guidance-scale', str(args.contact_guidance_scale),
            '--contact-timing-source', args.hhc_contact_timing_source,
        ]
        if args.hhc_contact_timing_file:
            command += [
                '--contact-timing-file',
                os.path.abspath(args.hhc_contact_timing_file),
            ]
        command.append('--contact-guidance' if args.contact_guidance else '--no-contact-guidance')
        return command + common, HHC_ROOT
    if args.method == 'intermask':
        return [
            sys.executable, 'tools_export_official_interx.py', '--gpu', str(args.gpu),
            '--length-policy', args.intermask_length_policy,
            '--time-steps', str(args.sampling_steps),
        ] + common, INTERMASK_ROOT
    if args.method == 'coshmdm':
        return [
            sys.executable, 'tools/export_official_interx.py',
            '--checkpoint', os.path.abspath(args.checkpoint), '--gpu', str(args.gpu),
        ] + common, COSHMDM_ROOT
    if args.method == 'timotion':
        return [
            sys.executable,
            os.path.join(
                HHC_ROOT, 'tools/export_timotion_official_interx_standard.py'
            ),
            '--checkpoint', os.path.abspath(args.checkpoint),
            '--gpu', str(args.gpu),
        ] + common, TIMOTION_ROOT
    return [
        os.path.abspath(args.interact2ar_python),
        os.path.join(HHC_ROOT, 'tools/export_interact2ar_official_interx.py'),
        '--checkpoint', os.path.abspath(args.checkpoint),
        '--repository-root', INTERACT2AR_ROOT,
        '--gpu', str(args.gpu),
    ] + common, INTERACT2AR_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--method',
        choices=('hhc', 'intermask', 'coshmdm', 'interact2ar', 'timotion'),
        required=True,
    )
    parser.add_argument('--checkpoint', help='Required except for InterMask')
    parser.add_argument(
        '--vq-checkpoint',
        default='checkpoints/vq_precleanup_raw12288/net_175000.pth',
    )
    parser.add_argument('--hhc-stats-dir', default='dataset/interx')
    parser.add_argument('--hhc-stats-prefix', default='Interx')
    parser.add_argument(
        '--hhc-contact-map', default=None,
        help=(
            'HHC train-only code-contact map. Defaults beside the selected '
            'VQ; augmented training arms must pass their immutable cache map.'
        ),
    )
    parser.add_argument(
        '--hhc-contact-map-contract',
        choices=('paper_formula_train_only', 'mmm_legacy_full_axis_bug'),
        default='paper_formula_train_only',
        help=(
            'Strict HHC contact-map provenance contract. The legacy option is '
            'for a disclosed one-replication implementation sensitivity only.'
        ),
    )
    parser.add_argument(
        '--hhc-vq-contract',
        choices=('current', 'original_8192', 'precleanup_raw12288'),
        default='precleanup_raw12288',
        help='HHC tokenizer schema passed to the official exporter.',
    )
    parser.add_argument(
        '--hhc-mgt-contract', choices=('current', 'historical_mmm'),
        default='current',
        help='HHC transformer architecture passed to the official exporter.',
    )
    parser.add_argument(
        '--hhc-mgt-shape', action=argparse.BooleanOptionalAction, default=True,
        help=(
            'Whether the selected HHC transformer checkpoint uses beta/gender '
            'conditioning. The VQ decoder remains shape-conditioned.'
        ),
    )
    parser.add_argument(
        '--hhc-inference-conditioning-contract',
        choices=(
            'training_contract', 'paper_text_cfg',
            'historical_mmm_shape_dropout',
        ),
        default='training_contract',
        help=(
            'Inference-only CFG contract. Set paper_text_cfg for the frozen '
            'pre-cleanup historical-MGT common-50 protocol.'
        ),
    )
    parser.add_argument(
        '--selection-manifest', default=None,
        help='HHC-only validation-FID checkpoint-selection manifest.',
    )
    parser.add_argument('--requests-dir', default='output/fair_interx_requests')
    parser.add_argument('--requests-manifest', default=None)
    parser.add_argument('--output-root', default='output/fair_eval')
    parser.add_argument('--label', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--sampling-steps', type=int, default=50,
        help='Common HHC/InterMask iterative-sampling budget for the fair table.',
    )
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--cfg-scale', type=float, default=3.5)
    parser.add_argument('--contact-guidance-scale', type=float, default=1.0)
    parser.add_argument(
        '--hhc-contact-timing-source',
        choices=('none', 'request_oracle', 'full_valid_weak', 'external_text_predictor'),
        default='request_oracle',
    )
    parser.add_argument('--hhc-contact-timing-file')
    parser.add_argument(
        '--interact2ar-python',
        default=os.path.join(INTERACT2AR_ROOT, '.venv/bin/python'),
        help='Python from the official Interact2AR uv-locked environment.',
    )
    parser.add_argument('--start-replication', type=int, default=0)
    parser.add_argument('--end-replication', type=int, default=20)
    parser.add_argument(
        '--shard-manifest', action='store_true',
        help=(
            'Write run_manifest_START_END.json for a strict sub-range so '
            'disjoint GPU workers never race on run_manifest.json. Merge the '
            'shard manifests after all 20 replications complete.'
        ),
    )
    parser.add_argument('--contact-guidance', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--intermask-length-policy', choices=('exact', 'author-floor'), default='exact'
    )
    parser.add_argument('--force-score', action='store_true')
    parser.add_argument(
        '--multimodality-audit', action=argparse.BooleanOptionalAction, default=True,
        help='Compute all 30 choose 2 embedding distances for every MM prompt.',
    )
    parser.add_argument(
        '--manifest-only', action='store_true',
        help=(
            'write provenance only for a new empty output directory; existing '
            'generations cannot be rebound to newer code'
        ),
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.method == 'hhc' and args.hhc_contact_map is None:
        args.hhc_contact_map = os.path.join(
            os.path.dirname(os.path.abspath(args.vq_checkpoint)),
            'code_contact_probs.npy',
        )

    if args.method in ('hhc', 'coshmdm', 'interact2ar', 'timotion') and not args.checkpoint:
        parser.error(f'--checkpoint is required for {args.method}')
    if args.selection_manifest and args.method not in ('hhc', 'timotion'):
        parser.error('--selection-manifest is only valid for HHC or TiMotion')
    if (args.hhc_contact_timing_source == 'external_text_predictor'
            and not args.hhc_contact_timing_file):
        parser.error(
            '--hhc-contact-timing-file is required for external_text_predictor'
        )
    if args.method == 'hhc':
        if args.contact_guidance and args.hhc_contact_timing_source == 'none':
            parser.error('contact guidance requires an explicit non-none timing source')
        if (not args.contact_guidance
                and args.hhc_contact_timing_source != 'none'):
            parser.error(
                'no-contact-guidance must explicitly use '
                '--hhc-contact-timing-source none'
            )
        if args.hhc_contact_map_contract == 'mmm_legacy_full_axis_bug':
            if not args.contact_guidance:
                parser.error('legacy contact-map sensitivity requires contact guidance')
            if (args.start_replication, args.end_replication) != (0, 1):
                parser.error(
                    'legacy contact-map sensitivity is restricted to replication 0'
                )
    if not 0 <= args.start_replication < args.end_replication <= 20:
        parser.error('replication range must satisfy 0 <= start < end <= 20')
    if args.shard_manifest and (args.start_replication, args.end_replication) == (0, 20):
        parser.error('--shard-manifest is only valid for a strict sub-range')
    if args.method == 'interact2ar' and not os.path.isfile(args.interact2ar_python):
        parser.error(
            'Interact2AR environment is absent; run `uv sync --frozen` in '
            f'{INTERACT2AR_ROOT}'
        )

    label = args.label or args.method
    method_output = os.path.abspath(os.path.join(args.output_root, label))
    os.makedirs(method_output, exist_ok=True)
    requests_dir = os.path.abspath(args.requests_dir)
    request_sample_counts = []
    for replication in range(args.start_replication, args.end_replication):
        request_path = os.path.join(
            requests_dir, f'replication_{replication:02d}.npz'
        )
        if not os.path.isfile(request_path):
            raise FileNotFoundError(request_path)
        with np.load(request_path, allow_pickle=False) as frozen_request:
            request_sample_counts.append(int(len(frozen_request['names'])))
    if len(set(request_sample_counts)) != 1:
        raise ValueError(
            'all replications in one method run must have the same frozen '
            f'sample count, got {request_sample_counts}'
        )
    expected_samples = request_sample_counts[0]
    if expected_samples <= 0 or expected_samples % args.batch_size:
        raise ValueError(
            f'frozen sample count {expected_samples} is not a positive complete '
            f'batch for batch size {args.batch_size}'
        )
    if args.manifest_only:
        existing_generations = sorted(
            path for path in os.listdir(method_output)
            if path.endswith('_main.h5') or path.endswith('_mm.h5')
        )
        if existing_generations:
            raise ValueError(
                '--manifest-only cannot rebind existing generations; use '
                'freeze_fair_manifest_implementation_sources.py on their '
                f'original manifest instead: {existing_generations[:4]}'
            )
    environment = os.environ.copy()
    if args.method == 'coshmdm':
        environment['COSHMDM_STATS_PREFIX'] = 'interx_global'
    elif args.method == 'timotion':
        environment['TIMOTION_STATS_PREFIX'] = 'interx_global'

    if args.method == 'coshmdm':
        run([
            sys.executable, 'tools/audit_coshmdm_generation_contract.py',
            '--requests-manifest', (
                args.requests_manifest
                if args.requests_manifest
                else os.path.join(requests_dir, 'manifest.json')
            ),
            '--output', 'output/coshmdm_generation_contract_audit.json',
        ], HHC_ROOT, environment, args.dry_run)
    elif args.method == 'timotion':
        run([
            sys.executable, 'tools/audit_timotion_generation_contract.py',
            '--requests-manifest', (
                args.requests_manifest
                if args.requests_manifest
                else os.path.join(requests_dir, 'manifest.json')
            ),
            '--output', 'output/timotion_generation_contract_audit.json',
        ], HHC_ROOT, environment, args.dry_run)

    manifest = {
        'method': args.method,
        'label': label,
        'checkpoint': os.path.abspath(args.checkpoint) if args.checkpoint else None,
        'vq_checkpoint': os.path.abspath(args.vq_checkpoint) if args.method == 'hhc' else None,
        'hhc_stats_dir': (
            os.path.abspath(args.hhc_stats_dir) if args.method == 'hhc' else None
        ),
        'hhc_stats_prefix': (
            args.hhc_stats_prefix if args.method == 'hhc' else None
        ),
        'hhc_vq_contract': (
            args.hhc_vq_contract if args.method == 'hhc' else None
        ),
        'hhc_mgt_contract': (
            args.hhc_mgt_contract if args.method == 'hhc' else None
        ),
        'hhc_mgt_shape_conditioning': (
            bool(args.hhc_mgt_shape) if args.method == 'hhc' else None
        ),
        'hhc_inference_conditioning_contract': (
            args.hhc_inference_conditioning_contract
            if args.method == 'hhc' else None
        ),
        'hhc_contact_map': (
            os.path.abspath(args.hhc_contact_map)
            if args.method == 'hhc' else None
        ),
        'hhc_contact_map_contract': (
            args.hhc_contact_map_contract if args.method == 'hhc' else None
        ),
        'selection_manifest': (
            os.path.abspath(args.selection_manifest) if args.selection_manifest else None
        ),
        'requests_dir': requests_dir,
        'requests_manifest': (
            os.path.abspath(args.requests_manifest)
            if args.requests_manifest else os.path.join(requests_dir, 'manifest.json')
        ),
        'replication_range': [args.start_replication, args.end_replication],
        'samples_per_replication': expected_samples,
        'main_seed': 'request.seed',
        'mm_seed': 'request.seed + 100000',
        'per_repeat_seed': 'export_seed + repeat_index',
        'sampling_steps': int(args.sampling_steps),
        'temperature': args.temperature if args.method == 'hhc' else None,
        'cfg_scale': args.cfg_scale if args.method == 'hhc' else None,
        'contact_guidance_scale': (
            args.contact_guidance_scale if args.method == 'hhc' else None
        ),
        'contact_timing_source': (
            args.hhc_contact_timing_source
            if args.method == 'hhc' and args.contact_guidance
            else ('none' if args.method == 'hhc' else None)
        ),
        'contact_timing_source_requested': (
            args.hhc_contact_timing_source if args.method == 'hhc' else None
        ),
        'contact_timing_file': (
            os.path.abspath(args.hhc_contact_timing_file)
            if (args.method == 'hhc' and args.contact_guidance
                and args.hhc_contact_timing_file) else None
        ),
        'multimodality_audit_all_pairs': bool(args.multimodality_audit),
        'contact_guidance': args.contact_guidance if args.method == 'hhc' else None,
        'intermask_length_policy': (
            args.intermask_length_policy if args.method == 'intermask' else None
        ),
        'coshmdm_length_policy': (
            'exact_request_length_grouped' if args.method == 'coshmdm' else None
        ),
        'coshmdm_sampling_process': (
            'DDIM50_eta0_CFG3.5' if args.method == 'coshmdm' else None
        ),
        'timotion_length_policy': (
            'exact_request_length_grouped' if args.method == 'timotion' else None
        ),
        'timotion_first_frame_conditioning': (
            'none' if args.method == 'timotion' else None
        ),
        'timotion_sampling_process': (
            'DDIM50_eta0_CFG3.5' if args.method == 'timotion' else None
        ),
        'timotion_rotation6d_contract': (
            'standard_row_concatenated_v1' if args.method == 'timotion' else None
        ),
        'interact2ar_length_policy': (
            'exact_request_length_grouped' if args.method == 'interact2ar' else None
        ),
        'interact2ar_first_frame_conditioning': (
            'none' if args.method == 'interact2ar' else None
        ),
    }
    if not args.dry_run:
        manifest['provenance'] = provenance(args, method_output)
    manifest_name = (
        f'run_manifest_{args.start_replication:02d}_{args.end_replication:02d}.json'
        if args.shard_manifest else 'run_manifest.json'
    )
    manifest_path = os.path.join(method_output, manifest_name)
    if not args.dry_run:
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    if args.manifest_only:
        print(json.dumps({'manifest': manifest_path, 'status': 'written'}), flush=True)
        return

    for replication in range(args.start_replication, args.end_replication):
        request = os.path.join(requests_dir, f'replication_{replication:02d}.npz')
        if not os.path.isfile(request):
            raise FileNotFoundError(request)
        frozen = np.load(request, allow_pickle=False)
        seed = int(frozen['seed'])
        main_h5 = os.path.join(method_output, f'replication_{replication:02d}_main.h5')
        mm_h5 = os.path.join(method_output, f'replication_{replication:02d}_mm.h5')
        metrics = os.path.join(method_output, f'replication_{replication:02d}.json')
        timing_path = os.path.join(
            method_output, f'replication_{replication:02d}_generation_timing.json'
        )
        generation_preexisting = {
            'main': os.path.isfile(main_h5), 'multimodality': os.path.isfile(mm_h5)
        }

        command, cwd = exporter_command(args, request, main_h5, seed, 1)
        main_timing = run(command, cwd, environment, args.dry_run)
        command, cwd = exporter_command(
            args, request, mm_h5, seed + 100_000, 30, indices_key='mm_indices'
        )
        mm_timing = run(command, cwd, environment, args.dry_run)
        if not args.dry_run:
            with np.load(request, allow_pickle=False) as frozen_timing:
                main_samples = len(frozen_timing['names'])
                mm_prompts = len(frozen_timing['mm_indices'])
            generation_hashes = {
                'main_generations_sha256': artifact_record(main_h5)['sha256'],
                'mm_generations_sha256': artifact_record(mm_h5)['sha256'],
            }
            preserve_existing_timing = False
            if os.path.isfile(timing_path):
                try:
                    with open(timing_path, encoding='utf-8') as handle:
                        previous_timing = json.load(handle)
                    preserve_existing_timing = all(
                        previous_timing.get(key) == value
                        for key, value in generation_hashes.items()
                    )
                except (OSError, ValueError, TypeError):
                    preserve_existing_timing = False
            if preserve_existing_timing:
                print(json.dumps({
                    'timing': timing_path,
                    'status': 'preserved matching original timing',
                }), flush=True)
            else:
                comparable = not any(generation_preexisting.values())
                timing_status = (
                    'fresh_complete' if comparable
                    else 'resume_only_not_cross_method_comparable'
                )
                timing = {
                    'schema': 'fair_interx_generation_wall_timing_v1',
                    'status': timing_status,
                    'timing_comparable': comparable,
                    'preexisting_before_invocation': generation_preexisting,
                    'boundary': (
                        'complete exporter subprocess wall time, including model '
                        'load and HDF5 output; evaluator/scorers excluded'
                    ),
                    'method': args.method,
                    'label': label,
                    'replication': replication,
                    'requests': os.path.abspath(request),
                    'requests_sha256': artifact_record(request)['sha256'],
                    'main_generations': os.path.abspath(main_h5),
                    'main_generations_sha256': generation_hashes[
                        'main_generations_sha256'
                    ],
                    'mm_generations': os.path.abspath(mm_h5),
                    'mm_generations_sha256': generation_hashes[
                        'mm_generations_sha256'
                    ],
                    'main': {
                        **main_timing,
                        'generated_samples': main_samples,
                        'seconds_per_sample_amortized': (
                            main_timing['elapsed_seconds'] / main_samples
                        ),
                    },
                    'multimodality': {
                        **mm_timing,
                        'prompts': mm_prompts,
                        'repeats_per_prompt': 30,
                        'generated_samples': mm_prompts * 30,
                        'seconds_per_sample_amortized': (
                            mm_timing['elapsed_seconds'] / (mm_prompts * 30)
                        ),
                    },
                }
                atomic_json(timing_path, timing)

        semantic_current = report_is_current(
            metrics,
            {
                'requests': request,
                'main_generations': main_h5,
                'mm_generations': mm_h5,
            },
            [request, main_h5, mm_h5],
        )
        if args.force_score or not semantic_current:
            run([
                sys.executable, 'tools/score_fair_interx_generations.py',
                '--requests', request, '--main-generations', main_h5,
                '--mm-generations', mm_h5, '--output', metrics,
                '--device', f'cuda:{args.gpu}', '--batch-size', '32',
            ], HHC_ROOT, environment, args.dry_run)
        else:
            print(json.dumps({'metrics': metrics, 'status': 'already exists'}), flush=True)

        if args.multimodality_audit:
            mm_audit = os.path.join(
                method_output,
                f'replication_{replication:02d}_multimodality_audit.json',
            )
            mm_audit_current = report_is_current(
                mm_audit,
                {'requests': request, 'mm_generations': mm_h5},
                [request, mm_h5],
            )
            if args.force_score or not mm_audit_current:
                run([
                    sys.executable, 'tools/analyze_fair_interx_multimodality.py',
                    '--requests', request, '--mm-generations', mm_h5,
                    '--output', mm_audit, '--device', f'cuda:{args.gpu}',
                    '--batch-size', '32',
                ], HHC_ROOT, environment, args.dry_run)
            else:
                print(json.dumps({
                    'multimodality_audit': mm_audit,
                    'status': 'already exists',
                }), flush=True)

        if replication == 0:
            physical = os.path.join(method_output, 'replication_00_physical.json')
            physical_current = report_is_current(
                physical,
                {'requests': request, 'generations': main_h5},
                [request, main_h5],
                expected_values={
                    'protocol': 'interx_official_v1_physical_joint_v1',
                    'definition': (
                        'LLM-free joint metrics using the frozen binary '
                        'contact schedule and subject-specific Inter-X '
                        'skeletons; penetration remains a separate mesh '
                        'metric.'
                    ),
                    'contact_threshold_m': 0.1,
                    'ground_height_threshold_m': 0.05,
                    'repeat': 0,
                },
            )
            if args.force_score or not physical_current:
                run([
                    sys.executable, 'tools/score_fair_interx_physical.py',
                    '--requests', request, '--generations', main_h5,
                    '--output', physical, '--device', f'cuda:{args.gpu}',
                ], HHC_ROOT, environment, args.dry_run)

    if args.start_replication == 0 and args.end_replication == 20:
        run([
            sys.executable, 'tools/aggregate_fair_interx_metrics.py',
            '--inputs', os.path.join(method_output, 'replication_[0-9][0-9].json'),
            '--output', os.path.join(method_output, 'summary.json'),
            '--expected-samples', str(expected_samples),
        ], HHC_ROOT, environment, args.dry_run)
        run([
            sys.executable,
            'tools/aggregate_fair_interx_generation_timings.py',
            '--inputs', os.path.join(
                method_output,
                'replication_[0-9][0-9]_generation_timing.json',
            ),
            '--output', os.path.join(
                method_output, 'generation_timing_summary.json'
            ),
        ], HHC_ROOT, environment, args.dry_run)
        if args.multimodality_audit:
            run([
                sys.executable,
                'tools/aggregate_fair_interx_multimodality_audits.py',
                '--metrics', os.path.join(
                    method_output, 'replication_[0-9][0-9].json'
                ),
                '--audits', os.path.join(
                    method_output,
                    'replication_[0-9][0-9]_multimodality_audit.json',
                ),
                '--output', os.path.join(
                    method_output, 'multimodality_audit_summary.json'
                ),
            ], HHC_ROOT, environment, args.dry_run)

    if not args.dry_run:
        validate_launch_sources_unchanged(manifest['provenance'])
        manifest['implementation_source_stability_verified_at_completion'] = True
        manifest['implementation_source_stability_verified_at_utc'] = (
            datetime.now(timezone.utc).isoformat()
        )
        atomic_json(manifest_path, manifest)


if __name__ == '__main__':
    main()
