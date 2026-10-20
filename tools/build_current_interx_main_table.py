#!/usr/bin/env python
"""Build the current strict Inter-X main table after all 20-rep runs exist."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('fid', 'matching', 'top1', 'top2', 'top3', 'diversity', 'multimodality')
METHODS = (
    ('HHC (no contact timing)', 'hhc_historical_plus_paper_mask_ratio_no_contact_common50'),
    ('HHC (oracle contact timing)', 'hhc_historical_plus_paper_mask_ratio_contact_common50'),
    ('InterMask (official ckpt, fair 50-step)', 'intermask_official_exact_common50'),
    ('Interact2AR (official ckpt, no GT first frame)', 'interact2ar_official_fair_no_gt_first_frame'),
    ('CoShMDM public-repository diffusion port (2-day cap)', 'coshmdm_interx_retrained_2daycap_seed123'),
    ('TiMotion public Transformer+LPA Inter-X port', 'timotion_transformer_lpa_interx_retrained_fair'),
)

CLAIM_SCOPE = {
    'hhc_historical_plus_paper_mask_ratio_no_contact_common50': 'HHC historical high-quality pipeline with paper mask-ratio contract; common 50-step fair protocol',
    'hhc_historical_plus_paper_mask_ratio_contact_common50': 'same HHC checkpoint with oracle GT contact timing; oracle ablation, not deployable main method',
    'intermask_official_exact_common50': 'official released Inter-X checkpoints under user-fixed common 50-step sampling; not the author-native 20-step paper row',
    'interact2ar_official_fair_no_gt_first_frame': 'official released checkpoint under unconditional frozen requests and legacy Inter-X evaluator; not the README GT-first-frame author protocol and not the new joint-evaluator paper row',
    'coshmdm_interx_retrained_2daycap_seed123': 'user-requested public-repository runnable 262D diffusion pipeline ported and retrained on Inter-X under the two-day cap; the release lacks the paper-described VQ, body-shape conditioning and RL-SDF pipeline and has no Inter-X checkpoint, so this is not a paper-architecture reproduction',
    'timotion_transformer_lpa_interx_retrained_fair': 'public Transformer+LPA architecture ported and retrained on Inter-X; public repository has no native Inter-X path or RWKV implementation, so not the paper-best RWKV row',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    if not path.is_file():
        raise ValueError(f'missing required artifact: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def finite_number(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f'{label} must be finite')
    return float(value)


def validate_summary(path):
    payload = read_json(path)
    if payload.get('replications') != 20:
        raise ValueError(f'{path}: expected exactly 20 replications')
    if payload.get('samples_per_replication') != 1696:
        raise ValueError(f'{path}: expected exactly 1696 samples per replication')
    metrics = payload.get('metrics', {})
    if not set(METRICS).issubset(metrics):
        raise ValueError(f'{path}: incomplete metric set')
    compact = {}
    for name in METRICS:
        record = metrics[name]
        values = record.get('values')
        if not isinstance(values, list) or len(values) != 20:
            raise ValueError(f'{path}: {name} must contain 20 values')
        values = [finite_number(value, f'{path}/{name}/values') for value in values]
        mean = finite_number(record.get('mean'), f'{path}/{name}/mean')
        ci95 = finite_number(record.get('ci95'), f'{path}/{name}/ci95')
        if not math.isclose(mean, sum(values) / 20, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f'{path}: {name} mean does not match values')
        compact[name] = {'mean': mean, 'ci95': ci95, 'values': values}
    return payload, compact


def validate_artifact_record(record, expected_path, label):
    expected_path = expected_path.resolve()
    if not isinstance(record, dict):
        raise ValueError(f'{label}: missing artifact record')
    if Path(record.get('path', '')).resolve() != expected_path:
        raise ValueError(f'{label}: artifact path changed')
    if record.get('sha256') != sha256_file(expected_path):
        raise ValueError(f'{label}: artifact hash is stale')
    if 'bytes' in record and record.get('bytes') != expected_path.stat().st_size:
        raise ValueError(f'{label}: artifact size is stale')


def validate_ordinary_only_audit(path, fair_root):
    payload = read_json(path)
    labels = [label for _, label in METHODS]
    if (
        payload.get('schema') != 'hhc_final_interx_ordinary_only_metric_audit_v1'
        or payload.get('status') != 'pass'
        or payload.get('metric_protocol')
        != 'official_interx_full_672d_ordinary_only'
        or set(payload.get('methods', {})) != set(labels)
        or payload.get('evaluator_runtime', {}).get('device_type') != 'cpu'
        or payload.get('evaluator_runtime', {}).get('requested_device') != 'cpu'
    ):
        raise ValueError(f'{path}: current ordinary-only evaluator audit mismatch')
    implementation = payload.get('implementation', {})
    for key, expected in (
        ('audit_and_rescore', ROOT / 'tools/rescore_final_interx_ordinary.py'),
        ('semantic_scorer', ROOT / 'tools/score_fair_interx_generations.py'),
        ('aggregator', ROOT / 'tools/aggregate_fair_interx_metrics.py'),
        ('multimodality_aggregator',
         ROOT / 'tools/aggregate_fair_interx_multimodality_audits.py'),
        ('evaluator_options', ROOT / 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'),
        ('evaluator_checkpoint', ROOT / 'checkpoints/hhi/text_mot_match/model/finest.tar'),
        ('evaluator_wrapper', ROOT / 'models/evaluator_wrapper.py'),
        ('evaluator_option_loader', ROOT / 'options/get_eval_option.py'),
    ):
        validate_artifact_record(implementation.get(key), expected, f'{path}/{key}')
    for label in labels:
        method = payload['methods'][label]
        directory = (fair_root / label).resolve()
        if (
            Path(method.get('directory', '')).resolve() != directory
            or method.get('replications') != 20
        ):
            raise ValueError(f'{path}: invalid ordinary-only method record for {label}')
        validate_artifact_record(
            method.get('summary'), directory / 'summary.json',
            f'{path}/{label}/summary',
        )
        records = method.get('records')
        if not isinstance(records, list) or len(records) != 20:
            raise ValueError(f'{path}: incomplete ordinary-only records for {label}')
        for replication, record in enumerate(records):
            if (
                record.get('replication') != replication
                or record.get('evaluator_provenance_verified') is not True
                or record.get('evaluator_runtime_verified') is not True
            ):
                raise ValueError(
                    f'{path}: invalid ordinary-only record {label}/{replication}'
                )
            validate_artifact_record(
                record.get('metrics'),
                directory / f'replication_{replication:02d}.json',
                f'{path}/{label}/{replication}',
            )
    return payload


def validate_paired(
    path, reference_label, reference_directory, comparators, fair_root
):
    payload = read_json(path)
    if payload.get('schema') != 'fair_interx_paired_method_comparison_v1':
        raise ValueError(f'{path}: unexpected paired-comparison schema')
    if payload.get('replications') != 20 or payload.get('replication_ids') != list(range(20)):
        raise ValueError(f'{path}: paired comparison must cover frozen replications 0..19')
    if payload.get('reference', {}).get('label') != reference_label:
        raise ValueError(f'{path}: unexpected reference label')
    if Path(payload.get('reference', {}).get('directory', '')).resolve() != (
        fair_root / reference_directory
    ):
        raise ValueError(f'{path}: paired reference directory changed')
    records = payload.get('comparators', {})
    if set(records) != set(comparators):
        raise ValueError(f'{path}: comparator labels changed')
    for comparator, record in records.items():
        if Path(record.get('directory', '')).resolve() != (
            fair_root / comparators[comparator]
        ):
            raise ValueError(f'{path}: paired comparator directory changed for {comparator}')
        metrics = record.get('metrics', {})
        expected = {
            'fid', 'matching', 'top1', 'top2', 'top3', 'multimodality',
            'diversity_absolute_error',
        }
        if set(metrics) != expected:
            raise ValueError(f'{path}: incomplete paired metric set for {comparator}')
        for name, metric in metrics.items():
            finite_number(metric.get('wilcoxon_two_sided_p'), f'{path}/{comparator}/{name}/p')
            finite_number(metric.get('holm_familywise_p'), f'{path}/{comparator}/{name}/holm_p')
            interval = metric.get('reference_advantage_bootstrap_95ci')
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f'{path}: invalid bootstrap interval')
            for value in interval:
                finite_number(value, f'{path}/{comparator}/{name}/bootstrap_ci')
    return payload


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def format_value(record):
    return f"{record['mean']:.4f} ± {record['ci95']:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fair-eval-root', type=Path, default=Path('output/fair_eval'))
    parser.add_argument('--paired-main', type=Path, required=True)
    parser.add_argument('--paired-timing', type=Path, required=True)
    parser.add_argument('--ordinary-only-audit', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-markdown', type=Path, required=True)
    args = parser.parse_args()

    root = args.fair_eval_root.resolve()
    ordinary_only_audit_path = args.ordinary_only_audit.resolve()
    validate_ordinary_only_audit(ordinary_only_audit_path, root)
    method_records = {}
    gt_diversity_values = None
    artifacts = {}
    for display, label in METHODS:
        directory = root / label
        summary_path = directory / 'summary.json'
        manifest_path = directory / 'run_manifest.json'
        raw_summary, metrics = validate_summary(summary_path)
        manifest = read_json(manifest_path)
        if manifest.get('label') != label:
            raise ValueError(f'{manifest_path}: run label mismatch')
        if manifest.get('samples_per_replication') != 1696:
            raise ValueError(f'{manifest_path}: sample-count contract changed')
        if manifest.get('replication_range') != [0, 20]:
            raise ValueError(f'{manifest_path}: expected frozen replication range [0, 20]')
        current_gt = raw_summary.get('metrics', {}).get('diversity_gt', {}).get('values')
        if not isinstance(current_gt, list) or len(current_gt) != 20:
            raise ValueError(f'{summary_path}: missing 20 GT-diversity values')
        current_gt = [finite_number(value, f'{summary_path}/diversity_gt') for value in current_gt]
        if gt_diversity_values is None:
            gt_diversity_values = current_gt
        elif current_gt != gt_diversity_values:
            raise ValueError('methods do not share identical frozen GT replications')
        method_records[label] = {
            'display_name': display,
            'claim_scope': CLAIM_SCOPE[label],
            'metrics': metrics,
        }
        artifacts[label] = {
            'summary': {'path': str(summary_path), 'sha256': sha256_file(summary_path)},
            'run_manifest': {'path': str(manifest_path), 'sha256': sha256_file(manifest_path)},
        }

    no_timing = METHODS[0][1]
    oracle_timing = METHODS[1][1]
    paired_main = validate_paired(
        args.paired_main.resolve(), 'hhc_no_timing', no_timing,
        {
            'intermask': METHODS[2][1],
            'interact2ar': METHODS[3][1],
            'coshmdm': METHODS[4][1],
            'timotion': METHODS[5][1],
        },
        root,
    )
    paired_timing = validate_paired(
        args.paired_timing.resolve(), 'hhc_oracle', oracle_timing,
        {'hhc_no_timing': no_timing},
        root,
    )
    implementation = Path(__file__).resolve()
    payload = {
        'schema': 'hhc_current_interx_main_table_v1',
        'status': 'complete',
        'protocol': {
            'dataset': 'Inter-X',
            'replications': 20,
            'samples_per_replication': 1696,
            'pairing': paired_main['pairing'],
            'statistical_test': paired_main['test'],
            'multiplicity_correction': paired_main['multiplicity_correction'],
            'sampling_note': 'HHC and InterMask use 50 outer sampling steps',
            'claim_scope_note': 'A shared evaluator makes rows numerically comparable, but does not turn public-code Inter-X ports or altered sampler/conditioning protocols into paper-exact reproductions.',
        },
        'methods': method_records,
        'paired_main': paired_main,
        'paired_oracle_timing_effect': paired_timing,
        'artifacts': {
            'methods': artifacts,
            'paired_main': {
                'path': str(args.paired_main.resolve()),
                'sha256': sha256_file(args.paired_main.resolve()),
            },
            'paired_timing': {
                'path': str(args.paired_timing.resolve()),
                'sha256': sha256_file(args.paired_timing.resolve()),
            },
            'ordinary_only_metric_audit': {
                'path': str(ordinary_only_audit_path),
                'sha256': sha256_file(ordinary_only_audit_path),
            },
            'implementation': {
                'path': str(implementation), 'sha256': sha256_file(implementation),
            },
        },
    }
    atomic_write(args.output_json.resolve(), json.dumps(payload, indent=2, sort_keys=True) + '\n')

    header = '| Method | FID ↓ | Matching ↓ | R@1 ↑ | R@2 ↑ | R@3 ↑ | Diversity | MM ↑ |\n'
    divider = '|---|---:|---:|---:|---:|---:|---:|---:|\n'
    rows = []
    for _, label in METHODS:
        record = method_records[label]
        metric = record['metrics']
        rows.append('| ' + ' | '.join([
            record['display_name'], format_value(metric['fid']),
            format_value(metric['matching']), format_value(metric['top1']),
            format_value(metric['top2']), format_value(metric['top3']),
            format_value(metric['diversity']), format_value(metric['multimodality']),
        ]) + ' |\n')
    markdown = (
        '# Current fair Inter-X main table\n\n'
        'All rows use 20 paired frozen-request replications with 1,696 test samples '
        'per replication. Values are mean ± 95% CI. Paired Wilcoxon tests use a '
        'Holm–Bonferroni family-wise correction; machine-readable statistics are in '
        f'`{args.output_json.resolve()}`. Shared requests/evaluator make these rows '
        'numerically comparable, but rows explicitly labelled as public-code ports or '
        'altered conditioning/sampling are not paper-exact reproductions.\n\n'
        + header + divider + ''.join(rows)
    )
    atomic_write(args.output_markdown.resolve(), markdown)
    print(json.dumps({'status': 'complete', 'json': str(args.output_json.resolve()),
                      'markdown': str(args.output_markdown.resolve())}, indent=2))


if __name__ == '__main__':
    main()
