#!/usr/bin/env python
"""Audit that every review concern has an honest experiment/status mapping."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


EXPECTED_REQUIREMENTS = {
    'optimization_contribution',
    'chi3d_quantitative_generalization',
    'mmodality_analysis',
    'llm_robustness',
    'contact_metric_fairness',
    'optimization_runtime',
    'timotion_qualitative',
    'figure_2_typo',
    'second_generation_dataset',
    'llm_parsing_accuracy',
    'foot_skating',
    'no_contact_timing',
    'weaker_llm_sensitivity',
    'interx_split_protocol',
    'unseen_shape_quantitative',
    'unseen_action_quantitative',
    'shape_qualitative_baselines',
    'contact_penetration_main_table',
    'user_study_values',
    'llm_version_and_hallucination',
}
EXPECTED_SOURCES = {
    'Meta-review', 'Reviewer L4xb', 'Reviewer ZyzM', 'Reviewer NbvC'
}
ALLOWED_STATUSES = {
    'complete',
    'active_queued',
    'implemented_waiting_final_generations',
    'deferred_after_interx',
    'cancelled_no_claim',
    'external_pending',
    'manuscript_pending',
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--matrix', type=Path,
        default=Path('configs/review_requirement_matrix.json'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/review_requirement_coverage.json'),
    )
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding='utf-8'))
    require(
        matrix.get('schema') == 'hhc_review_requirement_matrix_v1',
        'review requirement matrix schema changed',
    )
    source = matrix.get('source_review', {})
    source_path = Path(source.get('path', ''))
    require(source_path.is_file(), f'missing source review PDF: {source_path}')
    require(
        sha256_file(source_path) == source.get('sha256'),
        'source review PDF SHA-256 changed',
    )

    plan_path = Path(matrix.get('plan_path', ''))
    require(plan_path.is_file(), f'missing review plan: {plan_path}')
    plan = plan_path.read_text(encoding='utf-8')
    plan_sections = {
        int(match.group(1))
        for match in re.finditer(r'^\s*(\d+)\.\s+', plan, flags=re.MULTILINE)
    }
    require(plan_sections == set(range(1, 17)), (
        f'review plan must contain numbered experiments 1..16, got '
        f'{sorted(plan_sections)}'
    ))

    requirements = matrix.get('requirements', [])
    ids = [record.get('id') for record in requirements]
    require(len(ids) == len(set(ids)), 'duplicate review requirement IDs')
    require(set(ids) == EXPECTED_REQUIREMENTS, (
        f'review coverage changed: missing={sorted(EXPECTED_REQUIREMENTS-set(ids))}, '
        f'extra={sorted(set(ids)-EXPECTED_REQUIREMENTS)}'
    ))

    covered_sources = set()
    evidence_records = []
    for record in requirements:
        label = record['id']
        sources = set(record.get('sources', []))
        require(sources and sources <= EXPECTED_SOURCES, (
            f'{label}: invalid or missing review sources: {sorted(sources)}'
        ))
        covered_sources.update(sources)
        require(record.get('issue'), f'{label}: issue paraphrase is empty')
        sections = record.get('plan_sections', [])
        require(sections and set(sections) <= plan_sections, (
            f'{label}: invalid plan sections {sections}'
        ))
        status = record.get('status')
        require(status in ALLOWED_STATUSES, f'{label}: invalid status {status!r}')
        evidence = record.get('evidence', [])
        require(evidence, f'{label}: no evidence or executable artifact declared')
        for raw_path in evidence:
            path = Path(raw_path)
            require(path.is_file(), f'{label}: missing evidence file {path}')
            evidence_records.append({
                'requirement': label,
                'path': str(path.resolve()),
                'sha256': sha256_file(path),
            })

    require(covered_sources == EXPECTED_SOURCES, (
        f'not every review source is covered: {sorted(EXPECTED_SOURCES-covered_sources)}'
    ))
    statuses = {record['id']: record['status'] for record in requirements}
    require(
        statuses['unseen_action_quantitative'] == 'cancelled_no_claim'
        and statuses['unseen_shape_quantitative'] == 'active_queued',
        'user-directed shape augmentation/action no-claim policy changed',
    )
    require(
        statuses['chi3d_quantitative_generalization'] == 'deferred_after_interx'
        and statuses['second_generation_dataset'] == 'deferred_after_interx',
        'other-dataset work must remain explicitly deferred until Inter-X completes',
    )
    require(
        statuses['user_study_values'] == 'external_pending',
        'a real user study must not be represented as a completed machine experiment',
    )

    counts = Counter(statuses.values())
    completed = sorted(
        label for label, status in statuses.items() if status == 'complete'
    )
    current_interx_pending = sorted(
        label for label, status in statuses.items()
        if status in {
            'active_queued', 'implemented_waiting_final_generations',
            'manuscript_pending',
        }
    )
    deliberately_out_of_current_scope = sorted(
        label for label, status in statuses.items()
        if status in {
            'deferred_after_interx', 'cancelled_no_claim', 'external_pending'
        }
    )
    payload = {
        'schema': 'hhc_review_requirement_coverage_audit_v1',
        'status': 'pass',
        'coverage_complete': True,
        'experiments_complete': not current_interx_pending,
        'source_review': {
            'path': str(source_path.resolve()),
            'sha256': source['sha256'],
        },
        'matrix': {
            'path': str(args.matrix.resolve()),
            'sha256': sha256_file(args.matrix),
        },
        'plan': {
            'path': str(plan_path.resolve()),
            'sha256': sha256_file(plan_path),
            'numbered_sections': sorted(plan_sections),
        },
        'requirements': len(requirements),
        'review_sources': sorted(covered_sources),
        'status_counts': dict(sorted(counts.items())),
        'completed': completed,
        'current_interx_pending': current_interx_pending,
        'deliberately_out_of_current_scope': deliberately_out_of_current_scope,
        'records': requirements,
        'evidence': evidence_records,
        'implementation_sha256': sha256_file(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, args.output)
    print(json.dumps({
        'status': payload['status'],
        'requirements': payload['requirements'],
        'status_counts': payload['status_counts'],
        'current_interx_pending': payload['current_interx_pending'],
        'deliberately_out_of_current_scope': (
            payload['deliberately_out_of_current_scope']
        ),
        'output': str(args.output),
    }, indent=2))


if __name__ == '__main__':
    main()
