#!/usr/bin/env python
"""Freeze review experiments that still require unavailable external assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = path.resolve()
    if not path.is_file():
        return {'path': str(path), 'exists': False}
    return {
        'path': str(path), 'exists': True, 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--chi3d-audit', type=Path,
        default=Path('output/chi3d_asset_audit.json'),
    )
    parser.add_argument(
        '--optimization-readiness', type=Path,
        default=Path('output/optimization_ablation_readiness.json'),
    )
    parser.add_argument(
        '--interhuman-checkpoint-dir', type=Path,
        default=Path('/mnt/8T-HDD/wenning/InterMask/checkpoints/interhuman'),
    )
    parser.add_argument(
        '--interhuman-data-root', type=Path, action='append',
        dest='interhuman_data_roots',
    )
    parser.add_argument(
        '--prosepose-chi3d-root', type=Path,
        default=Path('/home/wenning/projects/prosepose/datasets'),
        help=(
            'Optional local ProsePose CHI3D cache.  Its files named test are '
            'audited explicitly so an internal s03 split cannot be mistaken '
            'for the unavailable official CHI3D test subject package.'
        ),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/review_external_prerequisites.json'),
    )
    parser.add_argument(
        '--user-study-manifest', type=Path,
        default=Path('output/user_study/manifest.json'),
    )
    args = parser.parse_args()

    data_roots = args.interhuman_data_roots or [
        Path('/mnt/8T-HDD/wenning/InterMask/data/InterHuman'),
        Path('/mnt/8T-HDD/wenning/TIMotion/data/InterHuman'),
        Path('/mnt/8T-HDD/wenning/HHC/dataset/interhuman'),
    ]
    data_root_records = []
    for root in data_roots:
        resolved = root.resolve()
        data_root_records.append({
            'path': str(resolved),
            'exists': resolved.is_dir(),
            'train_split_exists': (resolved / 'train.txt').is_file(),
            'val_split_exists': (resolved / 'val.txt').is_file(),
            'test_split_exists': (resolved / 'test.txt').is_file(),
        })
    interhuman_data_ready = any(
        record['exists']
        and record['train_split_exists']
        and record['val_split_exists']
        and record['test_split_exists']
        for record in data_root_records
    )

    checkpoint_dir = args.interhuman_checkpoint_dir.resolve()
    interhuman_checkpoints = sorted(checkpoint_dir.rglob('*.tar')) \
        if checkpoint_dir.is_dir() else []

    chi3d_path = args.chi3d_audit.resolve()
    if not chi3d_path.is_file():
        raise FileNotFoundError(chi3d_path)
    chi3d = json.loads(chi3d_path.read_text(encoding='utf-8'))
    if chi3d.get('status') != 'pass':
        raise ValueError('CHI3D asset audit is not pass')
    chi3d_official_test_ready = bool(
        chi3d.get('official_test_assets', {}).get(
            'usable_for_official_test_evaluation'
        )
    )

    prosepose_root = args.prosepose_chi3d_root.resolve()
    prosepose_contact_path = (
        prosepose_root / 'original/CHI3D/test/'
        'test_gpt4v_prompt2_custom_losses.json'
    )
    prosepose_optimization_path = (
        prosepose_root / 'processed/CHI3D/test_optimization.pkl'
    )
    prosepose_contact_subjects = []
    prosepose_contact_entries = None
    if prosepose_contact_path.is_file():
        prosepose_contacts = json.loads(
            prosepose_contact_path.read_text(encoding='utf-8')
        )
        if not isinstance(prosepose_contacts, dict):
            raise ValueError(
                f'expected object in {prosepose_contact_path}, got '
                f'{type(prosepose_contacts).__name__}'
            )
        prosepose_contact_entries = len(prosepose_contacts)
        prosepose_contact_subjects = sorted({
            str(key).split('_', 1)[0] for key in prosepose_contacts
        })
    prosepose_is_official_test = bool(
        prosepose_contact_entries
        and set(prosepose_contact_subjects).isdisjoint({'s02', 's03', 's04'})
    )
    if prosepose_is_official_test:
        raise ValueError(
            'ProsePose CHI3D cache no longer matches the audited training-'
            'subject-only classification; inspect it before enabling official '
            'test evaluation'
        )

    optimization_path = args.optimization_readiness.resolve()
    optimization = None
    if optimization_path.is_file():
        optimization = json.loads(
            optimization_path.read_text(encoding='utf-8')
        )
    optimization_inputs_ready = bool(
        optimization
        and optimization.get('schema') == 'hhc_optimization_ablation_readiness_v1'
        and optimization.get('ready') is True
    )
    # Contact annotation is an external-model prerequisite, while conversion
    # of the final HHC generation is an internal downstream dependency.  Do
    # not call the already-complete 1,696-sample Luna cache "not ready" merely
    # because the selected HHC checkpoint has not produced motions yet.
    strict_contact_ready = bool(
        optimization
        and optimization.get('schema')
        == 'hhc_optimization_ablation_readiness_v1'
        and optimization.get('expected_samples') == 1696
        and optimization.get('contact_pickle_files') == 1696
        and optimization.get('contact_metadata_files') == 1696
        and not optimization.get('contact_manifest_mismatches')
        and not optimization.get('contact_cache_audit_mismatches')
        and optimization.get('artifacts', {}).get(
            'contact_extraction_manifest', {}
        ).get('exists') is True
        and optimization.get('artifacts', {}).get(
            'contact_cache_audit', {}
        ).get('exists') is True
    )

    unresolved = []
    if not interhuman_data_ready:
        unresolved.append('interhuman_dataset_not_installed')
    if not chi3d_official_test_ready:
        unresolved.append('chi3d_official_test_package_not_installed')
    if not strict_contact_ready:
        unresolved.append('strict_llm_test_contact_cache_not_ready')
    user_study_path = args.user_study_manifest.resolve()
    user_study = None
    if user_study_path.is_file():
        user_study = json.loads(user_study_path.read_text(encoding='utf-8'))
    user_study_ready = bool(
        user_study
        and user_study.get('schema') == 'hhc_blinded_user_study_v1'
        and user_study.get('status') == 'complete'
        and int(user_study.get('participants', 0)) > 0
        and user_study.get('randomized') is True
        and user_study.get('blinded') is True
    )
    if not user_study_ready:
        unresolved.append('blinded_user_study_not_conducted')

    payload = {
        'schema': 'hhc_review_external_prerequisites_v1',
        'status': 'pass',
        'all_external_prerequisites_ready': not unresolved,
        'unresolved_external_prerequisites': unresolved,
        'interhuman': {
            'dataset_ready': interhuman_data_ready,
            'candidate_data_roots': data_root_records,
            'author_checkpoint_dir': str(checkpoint_dir),
            'author_checkpoint_files': [
                artifact(path) for path in interhuman_checkpoints
            ],
            'checkpoint_is_not_a_dataset_substitute': True,
            'execution_scope': (
                'after Inter-X completion, ingest and protocol-audit the '
                'results produced on the user-designated external server'
            ),
            'local_retraining_authorized': False,
            'baseline_checkpoint_policy': (
                'use compatible author checkpoints first; retrain a baseline '
                'only when its released checkpoint is unavailable or '
                'protocol/data incompatible'
            ),
            'required_experiment': (
                'after Inter-X completion, verify the external InterHuman '
                'split/checkpoint/evaluator provenance and merge its shared-'
                'protocol repeated-evaluation results; do not rerun the '
                'InterHuman baseline on this machine'
            ),
        },
        'chi3d': {
            'asset_audit': artifact(chi3d_path),
            'asset_classification': chi3d.get('asset_classification'),
            'local_train_samples': chi3d.get('counts', {}).get('train_samples'),
            'official_test_ready': chi3d_official_test_ready,
            'official_test_assets': chi3d.get('official_test_assets'),
            'allowed_local_protocol': chi3d.get('protocol_guardrail', {}).get(
                'allowed_local_generalization_protocol'
            ),
            'official_test_requirement': chi3d.get('protocol_guardrail', {}).get(
                'official_test_requirement'
            ),
            'prosepose_test_named_cache_audit': {
                'root': str(prosepose_root),
                'cache_detected': (
                    prosepose_contact_path.is_file()
                    or prosepose_optimization_path.is_file()
                ),
                'contact_records': artifact(prosepose_contact_path),
                'contact_record_count': prosepose_contact_entries,
                'contact_subjects': prosepose_contact_subjects,
                'processed_optimization_cache': artifact(
                    prosepose_optimization_path
                ),
                'usable_as_official_chi3d_test': prosepose_is_official_test,
                'classification': (
                    (
                        'internal_experiment_split_of_official_training_subject_s03; '
                        'filename_test_does_not_mean_official_chi3d_test_package'
                    ) if prosepose_contact_path.is_file() else
                    'cache_absent_not_official_chi3d_test_package'
                ),
            },
        },
        'strict_llm_contact_cache': {
            'ready': strict_contact_ready,
            'readiness_artifact': artifact(optimization_path),
            'samples': (
                optimization.get('contact_pickle_files')
                if optimization else 0
            ),
            'metadata_records': (
                optimization.get('contact_metadata_files')
                if optimization else 0
            ),
            'blockers': (
                [] if strict_contact_ready else
                ['strict_contact_cache_contract_incomplete']
            ),
        },
        'optimization_inputs': {
            'ready': optimization_inputs_ready,
            'classification': 'internal_downstream_dependency',
            'blockers': optimization.get('blockers', []) if optimization else [
                'optimization_readiness_artifact_missing'
            ],
        },
        'user_study': {
            'ready': user_study_ready,
            'manifest': artifact(user_study_path),
            'requirement': (
                'real participants, disclosed count/questions, randomized and '
                'blinded presentation, preference rates with confidence '
                'intervals and significance tests'
            ),
            'synthetic_responses_permitted': False,
        },
        'scope_note': (
            'These missing external assets do not invalidate the frozen '
            'Inter-X common-protocol main table, but they prevent claiming '
            'completion of the corresponding second-dataset, official-CHI3D, '
            'or real-participant user-study experiments. The strict Luna '
            'contact cache and the internally pending optimizer generation '
            'inputs are reported separately.'
        ),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
