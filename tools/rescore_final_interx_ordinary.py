#!/usr/bin/env python
"""Rescore reused baselines and freeze an ordinary-only final metric audit."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    'fid', 'diversity_gt', 'diversity', 'top1', 'top2', 'top3',
    'matching', 'multimodality',
)
FORBIDDEN_METRICS = {
    'fid_body_controlled', 'diversity_body_controlled',
    'diversity_gt_body_controlled',
}
FINAL_LABELS = (
    'hhc_formal_raw12288_repaired_mgt_common50_no_contact',
    'hhc_formal_raw12288_repaired_mgt_common50_contact',
    'intermask_official_exact_common50',
    'interact2ar_official_fair_no_gt_first_frame',
    'coshmdm_interx_retrained_2daycap_seed123',
    'timotion_transformer_lpa_interx_retrained_fair',
)
CURRENT_MAIN_LABELS = (
    'hhc_historical_plus_paper_mask_ratio_no_contact_common50',
    'hhc_historical_plus_paper_mask_ratio_contact_common50',
    'intermask_official_exact_common50',
    'interact2ar_official_fair_no_gt_first_frame',
    'coshmdm_interx_retrained_2daycap_seed123',
    'timotion_transformer_lpa_interx_retrained_fair',
)
ALLOWED_LABELS = tuple(dict.fromkeys(FINAL_LABELS + CURRENT_MAIN_LABELS))
DEFAULT_RESCORE_LABELS = FINAL_LABELS


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path), 'sha256': sha256_file(path),
        'bytes': path.stat().st_size,
    }


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def expected_evaluator_provenance():
    return {
        'contract': 'official_interx_full_672d',
        'options': artifact(ROOT / 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'),
        'checkpoint': artifact(
            ROOT / 'checkpoints/hhi/text_mot_match/model/finest.tar'
        ),
        'wrapper_source': artifact(ROOT / 'models/evaluator_wrapper.py'),
        'option_loader_source': artifact(ROOT / 'options/get_eval_option.py'),
        'scorer_source': artifact(
            ROOT / 'tools/score_fair_interx_generations.py'
        ),
    }


def expected_evaluator_runtime(device):
    requested = str(device)
    resolved = torch.device(requested)
    if resolved.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                f'final evaluator requested {requested}, but CUDA is unavailable'
            )
        index = (
            torch.cuda.current_device()
            if resolved.index is None else int(resolved.index)
        )
        device_name = torch.cuda.get_device_name(index)
    else:
        index = None
        device_name = 'cpu'
    return {
        'contract': 'pytorch_official_evaluator_single_device_v1',
        'requested_device': requested,
        'device_type': resolved.type,
        'device_index': index,
        'device_name': device_name,
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
    }


def replication_paths(fair_root, label, replication):
    directory = fair_root / label
    stem = f'replication_{replication:02d}'
    return {
        'requests': ROOT / 'output/fair_interx_requests' / f'{stem}.npz',
        'main': directory / f'{stem}_main.h5',
        'mm': directory / f'{stem}_mm.h5',
        'metrics': directory / f'{stem}.json',
    }


def rescore(fair_root, labels, device, batch_size):
    scorer = ROOT / 'tools/score_fair_interx_generations.py'
    aggregator = ROOT / 'tools/aggregate_fair_interx_metrics.py'
    multimodality_aggregator = (
        ROOT / 'tools/aggregate_fair_interx_multimodality_audits.py'
    )
    for label in labels:
        for replication in range(20):
            paths = replication_paths(fair_root, label, replication)
            for key in ('requests', 'main', 'mm'):
                if not paths[key].is_file():
                    raise FileNotFoundError(paths[key])
            subprocess.run(
                [
                    sys.executable, str(scorer),
                    '--requests', str(paths['requests']),
                    '--main-generations', str(paths['main']),
                    '--mm-generations', str(paths['mm']),
                    '--output', str(paths['metrics']),
                    '--device', device, '--batch-size', str(batch_size),
                ],
                cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
            )
            print(
                f'ordinary-only rescore complete: {label} '
                f'{replication + 1}/20',
                flush=True,
            )
        subprocess.run(
            [
                sys.executable, str(aggregator), '--inputs',
                str(fair_root / label / 'replication_[0-9][0-9].json'),
                '--output', str(fair_root / label / 'summary.json'),
                '--expected-samples', '1696',
            ],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )
        # The ordinary scorer also recomputes the official ten-pair
        # multimodality value.  Even CPU/GPU-equivalent embeddings can differ
        # by one float32 ULP, so the exhaustive 435-pair summary must be
        # re-aggregated from the newly written per-repeat metric records.
        # Otherwise its provenance is current but its official-values vector
        # is stale and the review report correctly rejects it.
        audit_paths = sorted(
            (fair_root / label).glob(
                'replication_[0-9][0-9]_multimodality_audit.json'
            )
        )
        if len(audit_paths) != 20:
            raise FileNotFoundError(
                f'{label}: expected 20 exhaustive multimodality audits, '
                f'found {len(audit_paths)}'
            )
        subprocess.run(
            [
                sys.executable, str(multimodality_aggregator),
                '--metrics', str(
                    fair_root / label / 'replication_[0-9][0-9].json'
                ),
                '--audits', str(
                    fair_root / label /
                    'replication_[0-9][0-9]_multimodality_audit.json'
                ),
                '--output', str(
                    fair_root / label / 'multimodality_audit_summary.json'
                ),
                '--expected-replications', '20',
            ],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )


def audit_label(
        fair_root, label, evaluator_provenance, evaluator_runtime,
        expected_replications=20):
    directory = (fair_root / label).resolve()
    records = []
    for replication in range(expected_replications):
        paths = replication_paths(fair_root, label, replication)
        metrics = json.loads(paths['metrics'].read_text(encoding='utf-8'))
        forbidden = sorted(FORBIDDEN_METRICS & set(metrics))
        missing = sorted(set(METRICS) - set(metrics))
        if forbidden or missing:
            raise ValueError(
                f'{paths["metrics"]}: forbidden={forbidden}, missing={missing}'
            )
        if (
            metrics.get('schema') != 'fair_interx_ordinary_metrics_v2'
            or metrics.get('status') != 'complete'
            or metrics.get('samples') != 1696
            or metrics.get('mm_samples') != 100
            or metrics.get('mm_repeats') != 30
            or metrics.get('generation_request_hash_verified') is not True
        ):
            raise ValueError(f'{paths["metrics"]}: metric contract mismatch')
        if metrics.get('evaluator_provenance') != evaluator_provenance:
            raise ValueError(
                f'{paths["metrics"]}: per-record evaluator provenance mismatch'
            )
        if metrics.get('evaluator_runtime') != evaluator_runtime:
            raise ValueError(
                f'{paths["metrics"]}: per-record evaluator runtime mismatch'
            )
        for key, record_key, hash_key in (
            ('requests', 'requests', 'requests_sha256'),
            ('main', 'main_generations', 'main_generations_sha256'),
            ('mm', 'mm_generations', 'mm_generations_sha256'),
        ):
            expected = paths[key].resolve()
            actual = Path(metrics.get(record_key, '')).resolve()
            if actual != expected or metrics.get(hash_key) != sha256_file(expected):
                raise ValueError(
                    f'{paths["metrics"]}: stale {record_key} provenance'
                )
        records.append({
            'replication': replication,
            'metrics': artifact(paths['metrics']),
            'requests_sha256': metrics['requests_sha256'],
            'main_generations_sha256': metrics['main_generations_sha256'],
            'mm_generations_sha256': metrics['mm_generations_sha256'],
            'evaluator_provenance_verified': True,
            'evaluator_runtime_verified': True,
        })

    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if (
        summary.get('replications') != expected_replications
        or summary.get('samples_per_replication') != 1696
        or set(summary.get('metrics', {})) != set(METRICS)
    ):
        raise ValueError(f'{summary_path}: ordinary-only summary mismatch')
    expected_inputs = [
        str(replication_paths(fair_root, label, replication)['metrics'].resolve())
        for replication in range(expected_replications)
    ]
    if summary.get('inputs') != expected_inputs:
        raise ValueError(f'{summary_path}: replication input order mismatch')
    return {
        'directory': str(directory),
        'replications': expected_replications,
        'summary': artifact(summary_path),
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--fair-root', type=Path, default=ROOT / 'output/fair_eval'
    )
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--rescore-label', action='append', dest='rescore_labels',
        choices=ALLOWED_LABELS,
    )
    parser.add_argument(
        '--audit-label', action='append', dest='audit_labels',
        choices=ALLOWED_LABELS,
    )
    parser.add_argument('--skip-rescore', action='store_true')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'output/fair_eval/final_ordinary_only_metric_audit.json',
    )
    args = parser.parse_args()
    fair_root = args.fair_root.resolve()
    rescore_labels = tuple(args.rescore_labels or DEFAULT_RESCORE_LABELS)
    audit_labels = tuple(args.audit_labels or FINAL_LABELS)
    if not args.skip_rescore:
        rescore(fair_root, rescore_labels, args.device, args.batch_size)
    evaluator_provenance = expected_evaluator_provenance()
    evaluator_runtime = expected_evaluator_runtime(args.device)
    methods = {
        label: audit_label(
            fair_root, label, evaluator_provenance, evaluator_runtime
        )
        for label in audit_labels
    }
    payload = {
        'schema': 'hhc_final_interx_ordinary_only_metric_audit_v1',
        'status': 'pass',
        'metric_protocol': 'official_interx_full_672d_ordinary_only',
        'metrics': list(METRICS),
        'forbidden_metrics': sorted(FORBIDDEN_METRICS),
        'rescored_labels': [] if args.skip_rescore else list(rescore_labels),
        'per_record_evaluator_provenance': evaluator_provenance,
        'evaluator_runtime': evaluator_runtime,
        'methods': methods,
        'implementation': {
            'audit_and_rescore': artifact(Path(__file__)),
            'semantic_scorer': artifact(
                ROOT / 'tools/score_fair_interx_generations.py'
            ),
            'aggregator': artifact(
                ROOT / 'tools/aggregate_fair_interx_metrics.py'
            ),
            'multimodality_aggregator': artifact(
                ROOT / 'tools/aggregate_fair_interx_multimodality_audits.py'
            ),
            'evaluator_options': artifact(
                ROOT / 'checkpoints/hhi/Comp_v6_KLD01/opt.txt'
            ),
            'evaluator_checkpoint': artifact(
                ROOT / 'checkpoints/hhi/text_mot_match/model/finest.tar'
            ),
            'evaluator_wrapper': artifact(
                ROOT / 'models/evaluator_wrapper.py'
            ),
            'evaluator_option_loader': artifact(
                ROOT / 'options/get_eval_option.py'
            ),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps({
        'status': 'pass', 'output': str(args.output.resolve()),
        'methods': list(methods), 'rescored_labels': payload['rescored_labels'],
    }, indent=2))


if __name__ == '__main__':
    main()
