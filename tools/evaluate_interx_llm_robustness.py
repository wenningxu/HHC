#!/usr/bin/env python
"""Evaluate strict contact parsers on the frozen Inter-X robustness subset."""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_interx_contact_pairs import validate_record  # noqa: E402
from utils.paramUtil import CANDIDATES  # noqa: E402


SCHEMA = 'hhc_interx_llm_robustness_evaluation_v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(temporary, path)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0
    return {
        'true_positive': int(tp), 'false_positive': int(fp),
        'false_negative': int(fn), 'precision': precision,
        'recall': recall, 'f1': f1,
    }


def presence_metrics(predicted, target):
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    result = prf(
        int(np.sum(predicted & target)),
        int(np.sum(predicted & ~target)),
        int(np.sum(~predicted & target)),
    )
    result.update({
        'accuracy': float(np.mean(predicted == target)),
        'predicted_positive': int(predicted.sum()),
        'timing_positive': int(target.sum()),
        'samples': len(target),
    })
    return result


def canonical_pairs(contacts, unordered_people=False):
    pairs = set()
    for contact in contacts:
        pair = (contact['p1_part'], contact['p2_part'])
        if unordered_people:
            pair = tuple(sorted(pair))
        pairs.add(pair)
    return pairs


def agreement_metrics(predictions, references, indices):
    totals = {'ordered': [0, 0, 0], 'unordered': [0, 0, 0]}
    exact = {'ordered': 0, 'unordered': 0}
    for index in indices:
        for label, unordered in (('ordered', False), ('unordered', True)):
            predicted = canonical_pairs(predictions[index], unordered)
            reference = canonical_pairs(references[index], unordered)
            totals[label][0] += len(predicted & reference)
            totals[label][1] += len(predicted - reference)
            totals[label][2] += len(reference - predicted)
            exact[label] += int(predicted == reference)
    output = {}
    for label in ('ordered', 'unordered'):
        output[label] = prf(*totals[label])
        output[label]['sample_exact_match_rate'] = exact[label] / len(indices)
        output[label]['samples'] = len(indices)
    return output


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return {
        'total': float(values.sum()),
        'mean_per_sample': float(values.mean()),
        'median_per_sample': float(np.median(values)),
        'p95_per_sample': float(np.percentile(values, 95)),
        'samples': int(values.size),
    }


def failed_attempt_usage(failure):
    """Recover timing/token metadata from either direct or wrapped failures."""
    metadata = failure.get('cli_metadata') or {}
    wall_seconds = metadata.get('wall_seconds')
    tokens_used = metadata.get('tokens_used')
    message = failure.get('error_message')
    try:
        wrapper_payload = json.loads(message) if isinstance(message, str) else {}
    except json.JSONDecodeError:
        wrapper_payload = {}
    if wall_seconds is None:
        wall_seconds = wrapper_payload.get('wall_seconds')
    if tokens_used is None:
        tokens_used = wrapper_payload.get('tokens_used')
    underlying = wrapper_payload.get('underlying_error_message')
    try:
        underlying_payload = json.loads(underlying) \
            if isinstance(underlying, str) else {}
    except json.JSONDecodeError:
        underlying_payload = {}
    if tokens_used is None:
        tokens_used = underlying_payload.get('tokens_used')
    return wall_seconds, tokens_used


def load_source(label, directory, names, captions, request_indices=None):
    directory = directory.resolve()
    manifest_path = directory / '_extraction_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError(f'{label}: extraction manifest is not complete')
    valid_region_ids = {
        int(region_id) for ids in CANDIDATES.values() for region_id in ids
    }
    contacts_by_index = []
    metadata_artifacts = []
    selected_batch_paths = set()
    selected_samples_per_batch = {}
    for subset_index, (name, caption) in enumerate(zip(names, captions)):
        pair_path = directory / f'{name}.pkl'
        metadata_path = directory / '_metadata' / f'{name}.json'
        with pair_path.open('rb') as handle:
            contacts = pickle.load(handle)
        if not isinstance(contacts, list):
            raise ValueError(f'{label}/{name}: payload is not a list')
        errors = sorted({
            error for contact in contacts
            for error in validate_record(contact, valid_region_ids)
        })
        if errors:
            raise ValueError(f'{label}/{name}: invalid constraints {errors}')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if (
            metadata.get('sample_name') != name
            or metadata.get('caption') != caption
            or metadata.get('status') != 'success'
            or metadata.get('structured_contact_rows') != len(contacts)
        ):
            raise ValueError(f'{label}/{name}: metadata/caption mismatch')
        if request_indices is not None and metadata.get('request_index') != int(
            request_indices[subset_index]
        ):
            raise ValueError(f'{label}/{name}: request index mismatch')
        batch_path = metadata.get('batch_metadata')
        if batch_path:
            batch_path = str(Path(batch_path).resolve())
            if not Path(batch_path).is_file():
                raise FileNotFoundError(batch_path)
            if metadata.get('batch_metadata_sha256') != sha256_file(batch_path):
                raise ValueError(f'{label}/{name}: batch hash mismatch')
            selected_batch_paths.add(batch_path)
            selected_samples_per_batch[batch_path] = (
                selected_samples_per_batch.get(batch_path, 0) + 1
            )
        contacts_by_index.append(contacts)
        metadata_artifacts.append({
            'name': name, 'pair_sha256': sha256_file(pair_path),
            'metadata_sha256': sha256_file(metadata_path),
        })

    failed_attempts = 0
    retry_batches = 0
    sample_wall_seconds = []
    sample_tokens = []
    wall_timed_attempts = 0
    wall_untimed_attempts = 0
    token_reported_attempts = 0
    for batch_path in sorted(selected_batch_paths):
        batch = json.loads(Path(batch_path).read_text(encoding='utf-8'))
        failures = batch.get('attempt_failures', [])
        failed_attempts += len(failures)
        retry_batches += int(bool(failures) or int(batch.get('attempts', 0)) > 1)
        batch_size = len(batch.get('sample_names', []))
        selected_count = selected_samples_per_batch[batch_path]
        if batch_size < selected_count or batch_size < 1:
            raise ValueError(f'{label}: invalid batch/sample accounting')
        attempt_usage = []
        final_metadata = batch.get('cli_metadata') or {}
        attempt_usage.append((
            final_metadata.get('wall_seconds'),
            final_metadata.get('tokens_used'),
        ))
        attempt_usage.extend(failed_attempt_usage(failure) for failure in failures)
        batch_wall_seconds = 0.0
        batch_tokens = 0.0
        has_wall = False
        has_tokens = False
        for wall_seconds, tokens_used in attempt_usage:
            if wall_seconds is None:
                wall_untimed_attempts += 1
            else:
                wall_seconds = float(wall_seconds)
                if not math.isfinite(wall_seconds) or wall_seconds < 0:
                    raise ValueError(f'{label}: invalid invocation wall time')
                batch_wall_seconds += wall_seconds
                wall_timed_attempts += 1
                has_wall = True
            if tokens_used is not None:
                tokens_used = float(tokens_used)
                if not math.isfinite(tokens_used) or tokens_used < 0:
                    raise ValueError(f'{label}: invalid invocation token count')
                batch_tokens += tokens_used
                token_reported_attempts += 1
                has_tokens = True
        if has_wall:
            sample_wall_seconds.extend(
                [batch_wall_seconds / batch_size] * selected_count
            )
        if has_tokens:
            sample_tokens.extend([batch_tokens / batch_size] * selected_count)

    if selected_batch_paths:
        resource_usage = {
            'scope': (
                'selected-sample amortized share of each Codex CLI batch; '
                'failed attempts included when timing/token metadata exists'
            ),
            'wall_seconds': summarize_values(sample_wall_seconds),
            'tokens': summarize_values(sample_tokens),
            'wall_timed_attempts': wall_timed_attempts,
            'wall_untimed_attempts': wall_untimed_attempts,
            'token_reported_attempts': token_reported_attempts,
        }
    else:
        timing = manifest.get('inference_wall_timing')
        resource_usage = {
            'scope': timing.get('scope') if timing else None,
            'wall_seconds': ({
                'total': timing['total_seconds'],
                'mean_per_sample': timing['mean_seconds_per_sample'],
                'median_per_sample': timing['median_seconds_per_sample'],
                'p95_per_sample': timing['p95_seconds_per_sample'],
                'samples': len(names),
            } if timing else None),
            'tokens': None,
            'wall_timed_attempts': len(names) if timing else 0,
            'wall_untimed_attempts': 0,
            'token_reported_attempts': 0,
        }
    return {
        'contacts': contacts_by_index,
        'manifest': manifest,
        'manifest_artifact': artifact(manifest_path),
        'metadata_records_sha256': hashlib.sha256(json.dumps(
            metadata_artifacts, sort_keys=True, separators=(',', ':'),
        ).encode()).hexdigest(),
        'selected_batch_count': len(selected_batch_paths),
        'retry_batches': retry_batches,
        'failed_attempts': failed_attempts,
        'invalid_final_outputs': 0,
        'unexecutable_constraint_records': 0,
        'resource_usage': resource_usage,
    }


def validate_cache_audit(
    path, expected_caption_binding, expected_requests_sha256,
    expected_samples,
):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not (
        payload.get('schema') == 'hhc_interx_codex_contact_cache_audit_v1'
        and payload.get('status') == 'pass'
        and payload.get('complete') is True
        and payload.get('integrity_passed') is True
        and payload.get('caption_binding') == expected_caption_binding
        and payload.get('inputs', {}).get('requests_sha256')
        == expected_requests_sha256
        and payload.get('counts', {}).get('expected_samples')
        == expected_samples
    ):
        raise ValueError(f'invalid strict cache audit: {path}')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--requests', type=Path,
        default=Path('output/interx_llm_robustness/requests.npz'),
    )
    parser.add_argument(
        '--luna-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_codex_luna_strict_v3_test'),
    )
    parser.add_argument(
        '--luna-audit', type=Path,
        default=Path('output/interx_luna_test_frozen_cache_audit.json'),
    )
    parser.add_argument(
        '--reference-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_codex_terra_robustness_test'),
    )
    parser.add_argument(
        '--reference-audit', type=Path,
        default=Path('output/interx_terra_robustness_cache_audit.json'),
    )
    parser.add_argument(
        '--rule-dir', type=Path,
        default=Path('dataset/interx/contact_pairs_rule_robustness_test'),
    )
    parser.add_argument(
        '--timing-dir', type=Path, default=Path('dataset/interx/contact'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('output/interx_llm_robustness/evaluation.json'),
    )
    args = parser.parse_args()

    requests = args.requests.resolve()
    selection_manifest_path = requests.with_suffix(
        requests.suffix + '.manifest.json'
    )
    selection_manifest = json.loads(
        selection_manifest_path.read_text(encoding='utf-8')
    )
    if (
        selection_manifest.get('schema')
        != 'hhc_interx_llm_robustness_requests_v1'
        or selection_manifest.get('status') != 'complete'
        or selection_manifest.get('output_sha256') != sha256_file(requests)
    ):
        raise ValueError('robustness request selection is stale')
    with np.load(requests, allow_pickle=False) as request:
        names = request['names'].astype(str).tolist()
        captions = request['captions'].astype(str).tolist()
        strata = request['strata'].astype(str).tolist()
        source_indices = request['source_request_indices'].astype(int).tolist()
    if len(names) != 128 or len(names) != len(set(names)):
        raise ValueError('expected 128 unique robustness requests')

    luna_audit = validate_cache_audit(
        args.luna_audit.resolve(), 'frozen_fair_evaluation_requests',
        selection_manifest['source_requests_sha256'], 1696,
    )
    reference_audit = validate_cache_audit(
        args.reference_audit.resolve(), 'frozen_fair_evaluation_requests',
        sha256_file(requests), len(names),
    )
    luna = load_source(
        'luna', args.luna_dir, names, captions, source_indices,
    )
    reference = load_source(
        'reference', args.reference_dir, names, captions,
        list(range(len(names))),
    )
    rule = load_source('rule', args.rule_dir, names, captions)
    if luna['manifest'].get('requested_model') != 'gpt-5.6-luna':
        raise ValueError('primary cache is not GPT-5.6 Luna')
    if reference['manifest'].get('requested_model') != 'gpt-5.6-terra':
        raise ValueError('reference cache is not GPT-5.6 Terra')
    if rule['manifest'].get('provider') != 'deterministic_lexical_rule_baseline':
        raise ValueError('rule cache provider changed')

    timing_positive = []
    for name in names:
        timing_path = args.timing_dir.resolve() / f'{name}.npy'
        timing = np.load(timing_path, allow_pickle=False).reshape(-1)
        timing_positive.append(bool(np.any(timing)))

    sources = {'luna': luna, 'terra_reference': reference, 'rule': rule}
    for source_name in ('luna', 'terra_reference'):
        usage = sources[source_name]['resource_usage']
        if (
            usage['wall_seconds'] is None
            or usage['tokens'] is None
            or usage['wall_seconds']['samples'] != len(names)
            or usage['tokens']['samples'] != len(names)
            or usage['wall_untimed_attempts'] != 0
        ):
            raise ValueError(
                f'{source_name}: incomplete Codex CLI wall/token accounting'
            )
    if rule['resource_usage']['wall_seconds'] is None:
        raise ValueError('rule: missing deterministic parser wall timing')
    strata_indices = {
        stratum: [index for index, value in enumerate(strata) if value == stratum]
        for stratum in sorted(set(strata))
    }
    results = {}
    for label, source in sources.items():
        predicted_presence = [bool(value) for value in source['contacts']]
        result = {
            'contact_presence_vs_manual_timing': presence_metrics(
                predicted_presence, timing_positive,
            ),
            'contact_rows': int(sum(len(value) for value in source['contacts'])),
            'empty_samples': int(sum(not value for value in source['contacts'])),
            'invalid_final_outputs': source['invalid_final_outputs'],
            'unexecutable_constraint_records': source[
                'unexecutable_constraint_records'
            ],
            'selected_batch_count': source['selected_batch_count'],
            'retry_batches': source['retry_batches'],
            'failed_attempts': source['failed_attempts'],
            'resource_usage': source['resource_usage'],
            'by_stratum': {},
        }
        for stratum, indices in strata_indices.items():
            result['by_stratum'][stratum] = presence_metrics(
                [predicted_presence[index] for index in indices],
                [timing_positive[index] for index in indices],
            )
        if label != 'terra_reference':
            result['part_pair_agreement_vs_terra_reference'] = agreement_metrics(
                source['contacts'], reference['contacts'], range(len(names))
            )
        results[label] = result

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'schema': SCHEMA,
        'status': 'complete',
        'sample_count': len(names),
        'reference_interpretation': (
            'GPT-5.6 Terra is a stronger-model consistency reference, not '
            'human region ground truth'
        ),
        'oracle_scope': (
            'manual Inter-X timing supplies binary contact presence only; '
            'Inter-X has no human 75-region labels, so oracle part F1 is not claimed'
        ),
        'results': results,
        'stratum_counts': {
            label: len(indices) for label, indices in strata_indices.items()
        },
        'artifacts': {
            'requests': artifact(requests),
            'selection_manifest': artifact(selection_manifest_path),
            'luna_cache_audit': artifact(args.luna_audit),
            'reference_cache_audit': artifact(args.reference_audit),
            'luna_manifest': luna['manifest_artifact'],
            'reference_manifest': reference['manifest_artifact'],
            'rule_manifest': rule['manifest_artifact'],
            'implementation': artifact(Path(__file__)),
        },
    }
    if not all(math.isfinite(value) for source in results.values() for value in (
        source['contact_presence_vs_manual_timing']['precision'],
        source['contact_presence_vs_manual_timing']['recall'],
        source['contact_presence_vs_manual_timing']['f1'],
    )):
        raise ValueError('robustness metrics contain non-finite values')
    atomic_json(output, report)
    print(json.dumps({
        'output': str(output), 'sample_count': len(names),
        'presence_f1': {
            label: result['contact_presence_vs_manual_timing']['f1']
            for label, result in results.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
