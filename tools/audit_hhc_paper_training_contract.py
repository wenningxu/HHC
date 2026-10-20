#!/usr/bin/env python
"""Audit the active HHC run against claims explicitly stated in the paper.

The report deliberately separates paper-stated settings, user-requested
evaluation settings, disclosed historical-tokenizer exceptions, and settings
that the paper does not specify.  It must not turn an implementation default
into a claimed paper hyperparameter.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def token_cache_contract_mismatches(
        active, canonical, cache_fields, augmented,
        augmented_tokenizer_mean_sha256=None,
        augmented_tokenizer_std_sha256=None,
        augmented_tokenizer_implementation_sha256=None):
    """Compare the baseline cache exactly or validate an augmented overlay.

    Shape augmentation necessarily changes the train split and token files; it
    must not be compared byte-for-byte with the unaugmented historical cache.
    It still reuses the same VQ and raw statistics and must be internally
    complete under the deterministic historical-root tokenizer.
    """
    if not augmented:
        return {
            key: {
                'expected_historical_exact': canonical.get(key),
                'active': active.get(key),
            }
            for key in cache_fields
            if active.get(key) != canonical.get(key)
        }

    if (not augmented_tokenizer_mean_sha256
            or not augmented_tokenizer_std_sha256
            or not augmented_tokenizer_implementation_sha256):
        raise ValueError(
            'augmented cache validation requires historical root statistics'
        )
    expected = {
        'token_cache_contract': 'deterministic_historical_root_v1',
        'vq_checkpoint_sha256': canonical.get('vq_checkpoint_sha256'),
        'nb_code': canonical.get('nb_code'),
        'code_dim': canonical.get('code_dim'),
        'down_t': canonical.get('down_t'),
        'cache_seed': 123,
        'cache_crop_policy': 'source_single_single_double_per_sample_sha256_v1',
        'cache_scope': 'train+val',
        'cache_input_filter': 'missing_eligible_train_val_before_tokenization',
        'normalization_mean_sha256': augmented_tokenizer_mean_sha256,
        'normalization_std_sha256': augmented_tokenizer_std_sha256,
        'tokenizer_implementation_sha256': (
            augmented_tokenizer_implementation_sha256
        ),
        'contact_map_scope': 'train_only',
        'contact_map_estimator': 'empirical_frequency_unseen_neutral_0.5',
        'val_split_sha256': canonical.get('val_split_sha256'),
        'status': 'complete',
    }
    mismatches = {
        key: {'expected_augmented': value, 'active': active.get(key)}
        for key, value in expected.items()
        if active.get(key) != value
    }
    positive_integers = (
        'expected_samples', 'cached_code_samples',
        'cached_contact_samples', 'eligible_train_samples',
        'eligible_val_samples', 'contact_map_samples',
    )
    for key in positive_integers:
        value = active.get(key)
        if not isinstance(value, int) or value <= 0:
            mismatches[key] = {
                'expected_augmented': 'positive integer', 'active': value,
            }
    expected_samples = active.get('expected_samples')
    relational = {
        'cached_code_samples_equals_expected': (
            active.get('cached_code_samples') == expected_samples
        ),
        'cached_contact_samples_equals_expected': (
            active.get('cached_contact_samples') == expected_samples
        ),
        'eligible_train_plus_val_equals_expected': (
            isinstance(active.get('eligible_train_samples'), int)
            and isinstance(active.get('eligible_val_samples'), int)
            and active.get('eligible_train_samples')
            + active.get('eligible_val_samples') == expected_samples
        ),
        'contact_map_uses_augmented_train_only': (
            active.get('contact_map_samples')
            == active.get('eligible_train_samples')
        ),
        'train_split_differs_from_unaugmented': (
            active.get('train_split_sha256')
            != canonical.get('train_split_sha256')
        ),
        'cache_hashes_present': all(
            isinstance(active.get(key), str) and len(active[key]) == 64
            for key in (
                'code_cache_sha256', 'contact_cache_sha256',
                'contact_map_sha256', 'train_split_sha256',
            )
        ),
    }
    for key, valid in relational.items():
        if not valid:
            mismatches[key] = {
                'expected_augmented': True, 'active': valid,
            }
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--training-manifest', type=Path, required=True,
        help=(
            'Manifest selected from the repaired formal run. This is required '
            'so an obsolete pre-repair run can never be audited implicitly.'
        ),
    )
    parser.add_argument(
        '--source-snapshot-manifest', type=Path,
        help=(
            'Optional immutable launch-source snapshot created before a '
            'post-training source transition. Without it, audit the live tree.'
        ),
    )
    parser.add_argument(
        '--vq-manifest', type=Path,
        default=Path('checkpoints/vq_precleanup_raw12288/training_manifest.json'),
    )
    parser.add_argument(
        '--paper', type=Path,
        default=Path('HHC__Generating_Human_Human_Close_Interactions_with_Contacts.pdf'),
    )
    parser.add_argument(
        '--protocol-audit', type=Path,
        default=Path('output/interx_protocol_audit.json'),
    )
    parser.add_argument(
        '--historical-cache-provenance', type=Path,
        default=Path('output/hhc_mgt_token_cache_provenance_audit.json'),
    )
    parser.add_argument(
        '--historical-tokenizer-runtime-audit', type=Path,
        default=Path('output/historical_mgt_tokenizer_runtime_audit.json'),
    )
    parser.add_argument('--historical-repo', type=Path, default=Path('/home/wenning/projects/MMM'))
    parser.add_argument('--precleanup-commit', default='75814e5')
    parser.add_argument(
        '--output', type=Path,
        default=Path(
            'output/hhc_formal_raw12288_repaired_mgt_training_contract_audit.json'
        ),
    )
    args = parser.parse_args()

    training_path = args.training_manifest.resolve()
    vq_path = args.vq_manifest.resolve()
    paper_path = args.paper.resolve()
    protocol_path = args.protocol_audit.resolve()
    cache_provenance_path = args.historical_cache_provenance.resolve()
    tokenizer_runtime_path = args.historical_tokenizer_runtime_audit.resolve()
    training = json.loads(training_path.read_text(encoding='utf-8'))
    vq = json.loads(vq_path.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    cache_provenance = json.loads(
        cache_provenance_path.read_text(encoding='utf-8')
    )
    tokenizer_runtime = json.loads(
        tokenizer_runtime_path.read_text(encoding='utf-8')
    )
    configured = training['paper_configuration']
    augmented_training = bool(
        training.get('resolved_arguments', {}).get('augmentation_manifest')
    )

    token_cache_path = Path(training['paths']['token_cache_manifest']).resolve()
    token_cache = json.loads(token_cache_path.read_text(encoding='utf-8'))
    canonical_cache_path = Path(
        cache_provenance['import_manifest']['path']
    ).resolve()
    canonical_cache = json.loads(
        canonical_cache_path.read_text(encoding='utf-8')
    )
    cache_fields = (
        'token_cache_contract', 'vq_checkpoint_sha256', 'nb_code',
        'code_dim', 'down_t', 'cache_seed', 'cache_crop_policy',
        'cache_scope', 'cache_input_filter', 'normalization_mean_sha256',
        'normalization_std_sha256', 'historical_codebook_source',
        'historical_contact_source',
        'historical_codebook_expected_names_sha256',
        'historical_contact_expected_names_sha256', 'code_cache_sha256',
        'contact_cache_sha256', 'expected_samples',
        'cached_code_samples', 'cached_contact_samples',
        'eligible_train_samples', 'eligible_val_samples',
        'contact_map_scope', 'contact_map_estimator', 'contact_map_samples',
        'contact_map_sha256', 'train_split_sha256', 'val_split_sha256',
        'status',
    )
    token_cache_mismatches = token_cache_contract_mismatches(
        token_cache, canonical_cache, cache_fields, augmented_training,
        augmented_tokenizer_mean_sha256=sha256_file(
            ROOT / 'dataset/interx/Interx_Mean_root.npy'
        ),
        augmented_tokenizer_std_sha256=sha256_file(
            ROOT / 'dataset/interx/Interx_Std_root.npy'
        ),
        augmented_tokenizer_implementation_sha256=sha256_file(
            ROOT / 'dataset/dataset_interx_tokenize_historical_root.py'
        ),
    )
    cache_manifest_binding_mismatches = {}
    if token_cache != training.get('cache_manifest'):
        cache_manifest_binding_mismatches['embedded_manifest'] = (
            'training manifest cache_manifest differs from its recorded file'
        )
    actual_token_cache_sha256 = sha256_file(token_cache_path)
    if training.get('sha256', {}).get('token_cache_manifest') != actual_token_cache_sha256:
        cache_manifest_binding_mismatches['sha256'] = {
            'recorded': training.get('sha256', {}).get('token_cache_manifest'),
            'actual': actual_token_cache_sha256,
        }
    cache_provenance_valid = bool(
        cache_provenance.get('status') == 'pass'
        and cache_provenance.get('byte_exact_import', {}).get(
            'all_code_files_equal'
        ) is True
        and cache_provenance.get('byte_exact_import', {}).get(
            'all_contact_files_equal'
        ) is True
        and cache_provenance.get('byte_exact_import', {}).get(
            'expected_files'
        ) == 9664
    )
    tokenizer_runtime_valid = bool(
        tokenizer_runtime.get('status') == 'pass'
        and tokenizer_runtime.get('cache', {}).get('code', {}).get('files')
        == 11370
        and tokenizer_runtime.get('cache', {}).get('contact', {}).get('files')
        == 11370
        and tokenizer_runtime.get('interpretation', {}).get(
            'shape_overlay_rebuild_contract'
        ) == (
            'to_rot_6d_then_Interx_Mean_root_Std_root_without_prepare_interx; '
            'retain source single/single/double crop distribution with a '
            'sample-keyed deterministic RNG'
        )
        and all(tokenizer_runtime.get('checks', {}).values())
    )

    checkpoint_selection = protocol.get(
        'checkpoint_selection_validation', {}
    )
    expected_checkpoint_selection = {
        'batch_size': 32,
        'drop_last': True,
        'evaluated_entries': 544,
        'intermask_contract_matches': True,
        'official_interx_contract_matches': True,
    }
    checkpoint_selection_mismatches = {
        key: {'expected': expected, 'actual': checkpoint_selection.get(key)}
        for key, expected in expected_checkpoint_selection.items()
        if checkpoint_selection.get(key) != expected
    }
    loss_validation = training.get('loader_contract', {}).get(
        'validation', {}
    )
    loss_validation_mismatches = {
        key: {'expected': expected, 'actual': loss_validation.get(key)}
        for key, expected in {
            'batch_size': 96,
            'drop_last_batches': 5,
            'active_samples_at_minimum_20_tokens': 534,
        }.items()
        if loss_validation.get(key) != expected
    }

    # ECCV 2026 submission, supplementary Sec. 6.1, paper lines 478--490.
    stated = {
        'attention_heads': 8,
        'batch_size': 96,
        'blocks': 9,
        'latent_dimension': 512,
        'learning_rate': 2e-4,
        'optimizer': 'AdamW',
        'schedule': 'linear warm-up then cosine decay',
        'temperature': 0.5,
        'total_iterations': 300000,
        'warmup_iterations': 1000,
        'weight_decay': 1e-6,
        'cfg_scale': 3.5,
        'contact_guidance_scale': 1.0,
    }
    stated_mismatches = {
        key: {'paper': expected, 'active': configured.get(key)}
        for key, expected in stated.items()
        if configured.get(key) != expected
    }

    qualitative_expected = {
        'contact_as_training_input': False,
        'mask_schedule': 'paper_kinematic',
        'sampling_order': 'Root->Body->Hands with immediate within-step updates',
        'shape_conditioning': True,
    }
    qualitative_mismatches = {
        key: {'paper': expected, 'active': configured.get(key)}
        for key, expected in qualitative_expected.items()
        if configured.get(key) != expected
    }

    formal_expected = {
        'paper_architecture': True,
        'complete_paper_training_contract': True,
        'architecture_contract': 'paper_reconstructed_cross_self_adaln_9_blocks',
        'mask_schedule': 'paper_kinematic',
        'objective': 'paper_masked_only',
        'conditioning': 'paper_text_cfg',
        'training_validation_contract': 'paper_isolated_common',
        'effective_validation_sampler': 'sample_intra_cas',
    }
    formal_mismatches = {
        key: {'expected': expected, 'active': configured.get(key)}
        for key, expected in formal_expected.items()
        if configured.get(key) != expected
    }
    allowed_token_windows = {'original_mmm_38', 'paper_50'}
    if configured.get('token_length_contract') not in allowed_token_windows:
        formal_mismatches['token_length_contract'] = {
            'expected_one_of': sorted(allowed_token_windows),
            'active': configured.get('token_length_contract'),
        }

    expected_sources = {
        relative: expected_hash
        for relative, expected_hash in training['sha256'].items()
        if relative.endswith('.py')
    }
    source_snapshot_manifest = None
    snapshot_sources = None
    if args.source_snapshot_manifest is not None:
        snapshot_path = args.source_snapshot_manifest.resolve()
        source_snapshot_manifest = json.loads(
            snapshot_path.read_text(encoding='utf-8')
        )
        if (
            source_snapshot_manifest.get('schema')
            != 'hhc_mgt_launch_source_snapshot_v1'
            or source_snapshot_manifest.get('status') != 'pass'
            or source_snapshot_manifest.get('source_count') != 14
            or source_snapshot_manifest.get('training_manifest', {}).get('sha256')
            != sha256_file(training_path)
        ):
            raise ValueError('invalid MGT launch-source snapshot manifest')
        snapshot_sources = source_snapshot_manifest.get('sources', {})
        if set(snapshot_sources) != set(expected_sources):
            raise ValueError('launch-source snapshot set differs from training manifest')

    source_mismatches = {}
    source_records = {}
    for relative, expected_hash in expected_sources.items():
        if snapshot_sources is None:
            path = ROOT / relative
            recorded_hash = None
            recorded_bytes = None
        else:
            snapshot_record = snapshot_sources[relative]
            path = Path(snapshot_record.get('path', ''))
            recorded_hash = snapshot_record.get('sha256')
            recorded_bytes = snapshot_record.get('bytes')
        current_hash = sha256_file(path)
        source_records[relative] = {
            **artifact(path), 'launch_sha256': expected_hash,
        }
        if (
            current_hash != expected_hash
            or (recorded_hash is not None and recorded_hash != current_hash)
            or (recorded_bytes is not None and recorded_bytes != path.stat().st_size)
        ):
            source_mismatches[relative] = {
                'launch': expected_hash,
                'snapshot_record': recorded_hash,
                'current': current_hash,
            }

    # ``train_t2i_trans.py`` is the relevant two-person HHC entrypoint.  An
    # earlier version of this audit accidentally inspected
    # ``train_t2m_trans.py`` (the inherited single-person MMM entrypoint),
    # which incorrectly made the visible pre-cleanup path look FP32.
    historical_source_file = 'train_t2i_trans.py'
    historical_source = subprocess.check_output(
        [
            'git', 'show',
            f'{args.precleanup_commit}:{historical_source_file}',
        ],
        cwd=args.historical_repo.resolve(),
    )
    historical_optimizer_file = 'utils/utils_model.py'
    historical_optimizer_source = subprocess.check_output(
        [
            'git', 'show',
            f'{args.precleanup_commit}:{historical_optimizer_file}',
        ],
        cwd=args.historical_repo.resolve(),
    )
    amp_markers = [
        marker for marker in (
            b'autocast', b'GradScaler', b'precision=', b'bf16', b'bfloat16'
        ) if marker in historical_source
    ]
    if b'precision="bf16-mixed"' not in historical_source:
        raise ValueError(
            'pre-cleanup two-person training source no longer proves '
            'bf16-mixed precision'
        )
    if b'betas=(0.5, 0.9)' not in historical_optimizer_source:
        raise ValueError(
            'pre-cleanup optimizer source no longer proves AdamW betas '
            '(0.5, 0.9)'
        )
    for marker in (
        b'MultiStepLR', b'mask_bernoulli', b'args.pkeep',
        b'num_layers=args.num_layers', b'num_local_layer=args.num_local_layer',
    ):
        if marker not in historical_source:
            raise ValueError(
                f'pre-cleanup two-person source missing expected marker {marker!r}'
            )

    actual_vq = vq['frozen_architecture']
    vq_exceptions = {
        'codebook_size_per_part': {
            'paper': 8096,
            'active': actual_vq['codebook_size_per_part'],
        },
        'training_updates': {
            'paper': 150000,
            'active_imported_checkpoint': 175000,
        },
        'policy': (
            'user-selected byte-identical pre-cleanup raw-12288 tokenizer; '
            'accepted only as an explicitly disclosed historical-mainline exception'
        ),
    }
    user_overrides = {
        'sampling_steps': {
            'paper_text': 'not specified; Fig. 2 illustrates Step 10',
            'active': configured['sampling_steps'],
            'policy': 'user-fixed common 50 outer steps for HHC and InterMask',
        },
        'tokenizer': vq_exceptions,
    }
    unspecified = {
        'numeric_precision': {
            'paper_text': 'not specified',
            'active': training['source_derived_configuration']['precision'],
            'precleanup_visible_training_script': (
                'bf16-mixed in the relevant two-person train_t2i_trans.py path'
            ),
            'precleanup_source_is_exact_run_archive': False,
            'interpretation': (
                'the active and visible pre-cleanup two-person paths use the '
                'same precision, so BF16 is not an observed legacy-vs-paper '
                'trajectory difference'
            ),
        },
        'mask_count_discretization': configured['mask_count_discretization'],
        'adamw_betas': {
            'paper_text': 'not specified',
            'active': training['source_derived_configuration']['optimizer_betas'],
        },
    }
    status = 'pass_with_disclosed_exceptions'
    if (stated_mismatches or qualitative_mismatches or formal_mismatches
            or source_mismatches
            or token_cache_mismatches
            or cache_manifest_binding_mismatches
            or not cache_provenance_valid
            or not tokenizer_runtime_valid
            or protocol.get('status') != 'pass'
            or checkpoint_selection_mismatches
            or loss_validation_mismatches):
        status = 'fail'
    report = {
        'schema': 'hhc_paper_training_contract_audit_v1',
        'status': status,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'audit_script': artifact(__file__),
        'paper': {
            **artifact(paper_path),
            'citation': 'Supplementary Sec. 6.1, lines 478--490; Method Sec. 3.1',
        },
        'active_training_manifest': artifact(training_path),
        'active_vq_manifest': artifact(vq_path),
        'interx_protocol_audit': artifact(protocol_path),
        'historical_token_cache_provenance': {
            **artifact(cache_provenance_path),
            'valid': cache_provenance_valid,
        },
        'historical_tokenizer_runtime_audit': {
            **artifact(tokenizer_runtime_path),
            'valid': tokenizer_runtime_valid,
        },
        'historical_token_cache_contract': {
            'validation_mode': (
                'deterministic_augmented_overlay'
                if augmented_training else 'byte_exact_historical_baseline'
            ),
            'active_manifest': artifact(token_cache_path),
            'canonical_manifest': artifact(canonical_cache_path),
            'checked_fields': list(cache_fields),
            'field_mismatches': token_cache_mismatches,
            'binding_mismatches': cache_manifest_binding_mismatches,
        },
        'paper_stated_numeric_contract': stated,
        'paper_stated_numeric_mismatches': stated_mismatches,
        'paper_stated_qualitative_contract': qualitative_expected,
        'paper_stated_qualitative_mismatches': qualitative_mismatches,
        'formal_implementation_contract': formal_expected,
        'formal_implementation_mismatches': formal_mismatches,
        'user_requested_or_historical_exceptions': user_overrides,
        'paper_unspecified_implementation_choices': unspecified,
        'validation_metric_logging': {
            'val/FID': 'current milestone value',
            'val/Diversity': 'running best-so-far value',
            'val/TOP1': 'running best-so-far value',
            'val/TOP2': 'running best-so-far value',
            'val/TOP3': 'running best-so-far value',
            'val/Matching': 'running best-so-far value',
            'checkpoint_selection': 'minimum current-milestone val/FID only',
            'interpretation': (
                'Repeated non-FID W&B values do not imply identical generated '
                'motions; those keys intentionally log cumulative best values.'
            ),
        },
        'validation_loader_roles': {
            'checkpoint_selection_fid': {
                **checkpoint_selection,
                'role': 'official evaluator used for checkpoint selection',
                'contract_mismatches': checkpoint_selection_mismatches,
            },
            'lightning_token_loss': {
                **loss_validation,
                'evaluated_entries_after_drop_last': (
                    loss_validation.get('batch_size', 0)
                    * loss_validation.get('drop_last_batches', 0)
                ),
                'role': 'token-loss monitoring only',
                'used_for_checkpoint_fid': False,
                'contract_mismatches': loss_validation_mismatches,
            },
            'interpretation': (
                'batch 96 / 480 entries belongs only to the Lightning '
                'token-loss loader; checkpoint-selection FID uses the '
                'official batch-32 / 544-entry InterMask/Inter-X loader'
            ),
        },
        'launch_source_stability': {
            'status': 'pass' if not source_mismatches else 'fail',
            'scope': (
                'immutable_launch_snapshot'
                if source_snapshot_manifest is not None else 'live_source_tree'
            ),
            'snapshot_manifest': (
                artifact(args.source_snapshot_manifest.resolve())
                if source_snapshot_manifest is not None else None
            ),
            'checked_files': len(source_records),
            'mismatches': source_mismatches,
            'records': source_records,
        },
        'precleanup_visible_source': {
            'repository': str(args.historical_repo.resolve()),
            'commit': args.precleanup_commit,
            'file': historical_source_file,
            'sha256': hashlib.sha256(historical_source).hexdigest(),
            'precision_markers_found': [value.decode() for value in amp_markers],
            'precision': 'bf16-mixed',
            'optimizer_file': historical_optimizer_file,
            'optimizer_file_sha256': hashlib.sha256(
                historical_optimizer_source
            ).hexdigest(),
            'optimizer_betas': [0.5, 0.9],
            'exact_historical_wandb_run_source_archive_available': False,
        },
        'legacy_vs_active_differential': {
            'purpose': (
                'diagnostic only; the legacy low-FID run is not a fair '
                'paper-architecture comparison'
            ),
            'same_observed_setting': {
                'numeric_precision': 'bf16-mixed',
                'vq_checkpoint': 'byte-identical raw-12288/175k',
                'learning_rate': 2e-4,
                'weight_decay': 1e-6,
                'training_updates': 300000,
            },
            'different_settings': {
                'architecture': {
                    'legacy': (
                        '1024 dim, 16 heads; 7 base self-attention + 2 '
                        'word-cross blocks + 9 output-head self-attention '
                        'blocks under num_layers=9,num_local_layer=2'
                    ),
                    'active': (
                        '9 paper blocks, each word-cross then motion-self, '
                        '512 dim and 8 heads'
                    ),
                },
                'mask_and_loss': {
                    'legacy': (
                        'one cosine ratio over the flattened three-channel '
                        'stream, pkeep=0.5 random-token corruption, NLL over '
                        'all valid tokens'
                    ),
                    'active': (
                        'three independently sampled and sorted cosine '
                        'ratios, shared MASK only, NLL over masked tokens'
                    ),
                },
                'optimizer_betas': {
                    'legacy': [0.5, 0.9],
                    'active': training['source_derived_configuration'][
                        'optimizer_betas'
                    ],
                },
                'learning_rate_schedule': {
                    'legacy': 'MultiStepLR at 150k with gamma 0.05',
                    'active': '1k linear warm-up followed by cosine decay',
                },
                'batch_size': {'legacy': 64, 'active': 96},
                'seed': {'legacy': 114514, 'active': 123},
            },
            'causal_conclusion': (
                'the evidence rules out VQ reconstruction, evaluator/export '
                'identity, common-50 sampling, and BF16 as unique causes; it '
                'does not isolate one cause among the deliberately changed '
                'paper architecture, mask/loss, optimizer, scheduler, batch, '
                'and seed settings'
            ),
        },
        'interpretation': (
            'The active MGT matches every numeric and qualitative setting '
            'explicitly stated by the paper. The imported raw-12288 VQ and '
            '50-step common evaluation budget are disclosed exceptions. BF16 '
            'is not claimed as a paper setting.'
        ),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if status == 'fail':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
