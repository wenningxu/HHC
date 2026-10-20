import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.extract_val_contacts import canonical_sha256, sha256_file  # noqa: E402
from eval import extract_interx_contacts_codex as codex_base  # noqa: E402
from eval import extract_interx_contacts_codex_frozen_requests as frozen  # noqa: E402
from eval.extract_interx_contacts_codex_frozen_requests import (  # noqa: E402
    bind_official_split_count,
    enrich_metadata,
    invoke_with_timing,
    load_binding,
)
from eval.resume_interx_contacts_codex_preserve_failures import (  # noqa: E402
    merge_prior_failures,
)
from tools.evaluate_interx_llm_robustness import (  # noqa: E402
    agreement_metrics, presence_metrics,
)
from tools.predict_interx_contact_regions_rule import rule_rows  # noqa: E402
from tools.prepare_interx_llm_robustness_requests import (  # noqa: E402
    validate_or_write_npz,
)


AUDITOR = ROOT / 'tools/audit_interx_codex_contact_cache.py'
EXTRACTOR = ROOT / 'eval/extract_interx_contacts_codex.py'
PARSER = ROOT / 'optimization/text_2_regions.py'
REGIONS = ROOT / 'utils/paramUtil.py'
OUTPUT_SCHEMA = ROOT / 'configs/contact_annotation_batch_schema.json'
BINDING_WRITER = ROOT / 'eval/extract_interx_contacts_codex_frozen_requests.py'


def build_cache(root):
    split = root / 'test.txt'
    requests = root / 'replication_00.npz'
    text_dir = root / 'texts'
    contact_dir = root / 'contacts'
    metadata_dir = contact_dir / '_metadata'
    batches_dir = contact_dir / '_batches'
    text_dir.mkdir()
    metadata_dir.mkdir(parents=True)
    batches_dir.mkdir()
    split.write_text('sample_a\nsample_not_requested\n', encoding='utf-8')
    caption = 'Person 1 taps Person 2 on the shoulder.'
    np.savez_compressed(
        requests,
        names=np.asarray(['sample_a']),
        captions=np.asarray([caption]),
    )

    implementation = {
        'extractor': {
            'path': str(EXTRACTOR), 'sha256': sha256_file(EXTRACTOR),
        },
        'contact_parser': {
            'path': str(PARSER), 'sha256': sha256_file(PARSER),
        },
        'region_dictionary': {
            'path': str(REGIONS), 'sha256': sha256_file(REGIONS),
        },
    }
    contract_fields = {
        'schema': 'hhc_interx_contact_pairs_codex_strict_v3',
        'split': str(split.resolve()),
        'split_sha256': sha256_file(split),
        'split_count': 2,
        'selection_offset': 0,
        'selection_limit': None,
        'sample_count': 1,
        'selected_names_sha256': canonical_sha256(['sample_a']),
        'selected_captions_sha256': canonical_sha256([caption]),
        'caption_source': 'frozen_fair_evaluation_requests',
        'requests': str(requests.resolve()),
        'requests_sha256': sha256_file(requests),
        'request_names_sha256': canonical_sha256(['sample_a']),
        'request_captions_sha256': canonical_sha256([caption]),
        'provider': 'openai_codex_cli_chatgpt_login',
        'requested_model': 'gpt-5.6-luna',
        'reasoning_effort': 'low',
        'codex_cli_version': 'synthetic',
        'jsonschema_version': 'synthetic',
        'batch_size': 32,
        'max_retries': 2,
        'timeout_seconds': 600,
        'prompt_template_sha256': 'synthetic-prompt',
        'output_schema': str(OUTPUT_SCHEMA),
        'output_schema_sha256': sha256_file(OUTPUT_SCHEMA),
        'implementation': implementation,
        'strict_schema': True,
        'api_key_recorded': False,
        'authentication_secret_recorded': False,
    }
    contract_hash = canonical_sha256(contract_fields)
    manifest = {
        **contract_fields,
        'contract_sha256': contract_hash,
        'status': 'complete',
        'completed_count': 1,
        'completed_batches_this_run': 1,
        'total_tokens_used': 7,
    }
    (contact_dir / '_extraction_manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8'
    )

    batch_path = batches_dir / 'batch.json'
    batch = {
        'schema': 'hhc_codex_contact_batch_v1',
        'status': 'success',
        'sample_names': ['sample_a'],
        'contract_sha256': contract_hash,
        'attempts': 1,
        'attempt_failures': [],
        'cli_metadata': {'tokens_used': 7, 'wall_seconds': 2.0},
    }
    batch_path.write_text(json.dumps(batch), encoding='utf-8')
    metadata = {
        'schema': manifest['schema'],
        'sample_name': 'sample_a',
        'caption': caption,
        'caption_sha256': hashlib.sha256(caption.encode()).hexdigest(),
        'caption_source': manifest['caption_source'],
        'requests': manifest['requests'],
        'requests_sha256': manifest['requests_sha256'],
        'request_binding_writer': {
            'path': str(BINDING_WRITER),
            'sha256': sha256_file(BINDING_WRITER),
        },
        'request_index': 0,
        'provider': manifest['provider'],
        'requested_model': manifest['requested_model'],
        'response_model_version': manifest['requested_model'],
        'reasoning_effort': manifest['reasoning_effort'],
        'prompt_template_sha256': manifest['prompt_template_sha256'],
        'status': 'success',
        'structured_contact_rows': 0,
        'batch_metadata': str(batch_path.resolve()),
        'batch_metadata_sha256': sha256_file(batch_path),
        'api_key_recorded': False,
        'authentication_secret_recorded': False,
    }
    metadata_path = metadata_dir / 'sample_a.json'
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    with (contact_dir / 'sample_a.pkl').open('wb') as handle:
        pickle.dump([], handle)
    return split, requests, text_dir, contact_dir, metadata_path


def run_audit(split, requests, text_dir, contact_dir, output):
    return subprocess.run(
        [
            sys.executable, str(AUDITOR),
            '--pairs-dir', str(contact_dir),
            '--split', str(split),
            '--text-dir', str(text_dir),
            '--requests', str(requests),
            '--output', str(output),
        ],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_strict_cache_audit_accepts_exact_frozen_request_binding():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        split, requests, text_dir, contact_dir, _ = build_cache(root)
        output = root / 'audit.json'
        result = run_audit(
            split, requests, text_dir, contact_dir, output,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(output.read_text(encoding='utf-8'))
        assert report['status'] == 'pass'
        assert report['complete'] is True
        assert report['integrity_passed'] is True
        assert report['resource_usage']['wall_seconds']['mean_per_sample'] == 2.0
        assert report['resource_usage']['tokens']['mean_per_sample'] == 7.0


def test_strict_cache_audit_rejects_missing_per_sample_request_hash():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        split, requests, text_dir, contact_dir, metadata_path = build_cache(root)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.pop('requests_sha256')
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        output = root / 'audit.json'
        result = run_audit(
            split, requests, text_dir, contact_dir, output,
        )
        assert result.returncode != 0
        report = json.loads(output.read_text(encoding='utf-8'))
        assert report['status'] == 'fail'
        assert 'requests_sha256' in report['metadata_errors']['sample_a']


def test_frozen_request_writer_checks_indexed_name_and_caption():
    with tempfile.TemporaryDirectory() as directory:
        requests = Path(directory) / 'requests.npz'
        np.savez_compressed(
            requests,
            names=np.asarray(['sample_a']),
            captions=np.asarray(['exact selected caption']),
        )
        binding = load_binding(requests)
        payload = {
            'caption_source': 'frozen_fair_evaluation_requests',
            'request_index': 0,
            'sample_name': 'sample_a',
            'caption': 'exact selected caption',
        }
        enriched = enrich_metadata(payload, binding)
        assert enriched['requests_sha256'] == sha256_file(requests)
        assert enriched['request_binding_writer']['sha256'] == sha256_file(
            BINDING_WRITER
        )
        try:
            enrich_metadata(dict(payload, caption='wrong caption'), binding)
        except ValueError as error:
            assert 'indexed frozen request' in str(error)
        else:
            raise AssertionError('mismatched frozen caption was accepted')


def test_frozen_request_writer_preserves_official_and_request_counts():
    with tempfile.TemporaryDirectory() as directory:
        split = Path(directory) / 'test.txt'
        split.write_text(
            'sample_a\nsample_not_requested\n', encoding='utf-8',
        )
        records = bind_official_split_count(
            [{'sample_name': 'sample_a'}], split,
        )
        assert len(records) == 2
        selected = records[0:]
        assert isinstance(selected, list)
        assert len(selected) == 1
        assert selected[0]['sample_name'] == 'sample_a'


def test_frozen_request_extractor_and_auditor_keep_distinct_counts():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        split = root / 'test.txt'
        requests = root / 'replication_00.npz'
        text_dir = root / 'texts'
        contact_dir = root / 'contacts'
        audit_path = root / 'audit.json'
        text_dir.mkdir()
        split.write_text(
            'sample_a\nsample_not_requested\n', encoding='utf-8',
        )
        caption = 'Person 1 taps Person 2 on the shoulder.'
        np.savez_compressed(
            requests,
            names=np.asarray(['sample_a']),
            captions=np.asarray([caption]),
        )
        argv = [
            str(BINDING_WRITER),
            '--split-file', str(split),
            '--text-dir', str(text_dir),
            '--requests', str(requests),
            '--out-dir', str(contact_dir),
            '--output-schema', str(OUTPUT_SCHEMA),
            '--model', 'gpt-5.6-luna',
            '--reasoning-effort', 'low',
            '--batch-size', '32',
        ]
        response = json.dumps({
            'records': [{'sample_name': 'sample_a', 'contacts': []}],
        })
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(
            codex_base, 'codex_version', return_value='codex-cli synthetic',
        ), mock.patch.object(
            codex_base, 'invoke_codex',
            return_value=(response, {'session_id': 'test', 'tokens_used': 7}),
        ):
            frozen.main()

        manifest = json.loads(
            (contact_dir / '_extraction_manifest.json').read_text(
                encoding='utf-8'
            )
        )
        assert manifest['split_count'] == 2
        assert manifest['sample_count'] == 1
        assert manifest['completed_count'] == 1
        batch_path = next((contact_dir / '_batches').glob('*.json'))
        batch = json.loads(batch_path.read_text(encoding='utf-8'))
        assert batch['cli_metadata']['wall_seconds'] >= 0
        assert batch['cli_metadata']['wall_clock_scope'] \
            == 'single_codex_cli_attempt'
        result = run_audit(
            split, requests, text_dir, contact_dir, audit_path,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(audit_path.read_text(encoding='utf-8'))
        assert report['status'] == 'pass'
        assert report['counts']['expected_samples'] == 1


def test_resume_wrapper_preserves_all_failed_attempt_sidecars():
    with tempfile.TemporaryDirectory() as directory:
        batch_dir = Path(directory) / '_batches'
        batch_dir.mkdir()
        path = batch_dir / 'batch.json'
        identity = {
            'sample_names': ['sample_a'],
            'captions_sha256': 'captions',
            'prompt_sha256': 'prompt',
            'contract_sha256': 'contract',
        }
        path.write_text(json.dumps({
            **identity,
            'status': 'failed',
            'attempts': 3,
            'attempt_failures': [
                {'attempt': 1}, {'attempt': 2}, {'attempt': 3},
            ],
        }), encoding='utf-8')
        merged = merge_prior_failures(path, {
            **identity,
            'status': 'success',
            'attempts': 2,
            'attempt_failures': [{'attempt': 1}],
        })
        assert merged['attempts'] == 5
        assert len(merged['attempt_failures']) == 4
        assert merged['resume_provenance'][
            'preserved_prior_failed_attempts'
        ] == 3


def test_llm_robustness_metrics_distinguish_role_order_and_timing():
    timing = presence_metrics([True, True, False], [True, False, False])
    assert timing['precision'] == 0.5
    assert timing['recall'] == 1.0
    assert timing['accuracy'] == 2 / 3
    luna = [[{'p1_part': 'left arm', 'p2_part': 'back'}]]
    terra = [[{'p1_part': 'back', 'p2_part': 'left arm'}]]
    agreement = agreement_metrics(luna, terra, [0])
    assert agreement['ordered']['f1'] == 0.0
    assert agreement['unordered']['f1'] == 1.0


def test_rule_contact_parser_is_deterministic_and_fail_closed():
    rows, rule = rule_rows(
        'The first person hugs the second person and pats their back.'
    )
    assert rule == 'hug_template'
    assert len(rows) == 2
    assert {row['contact area'] for row in rows} == {'hug'}
    empty, empty_rule = rule_rows(
        'Two people face one another and wave without touching.'
    )
    assert empty == []
    assert empty_rule == 'explicitly_negated_contact'


def test_frozen_request_invocation_records_success_and_failure_time():
    with mock.patch.object(
        frozen.time, 'perf_counter', side_effect=[10.0, 12.5],
    ):
        response, metadata = invoke_with_timing(
            lambda args, prompt: ('{}', {'tokens_used': 3}), None, 'prompt',
        )
    assert response == '{}'
    assert metadata['tokens_used'] == 3
    assert metadata['wall_seconds'] == 2.5
    with mock.patch.object(
        frozen.time, 'perf_counter', side_effect=[20.0, 21.25],
    ):
        try:
            invoke_with_timing(
                lambda args, prompt: (_ for _ in ()).throw(ValueError('bad')),
                None, 'prompt',
            )
        except frozen.TimedInvocationError as error:
            payload = json.loads(str(error))
            assert payload['wall_seconds'] == 1.25
            assert payload['underlying_error_type'] == 'ValueError'
        else:
            raise AssertionError('timed invocation swallowed a failed attempt')


def test_robustness_request_archive_is_byte_stable_and_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'requests.npz'
        payload = {
            'names': np.asarray(['a', 'b']),
            'captions': np.asarray(['one', 'two']),
        }
        assert validate_or_write_npz(output, payload) is False
        original_hash = sha256_file(output)
        assert validate_or_write_npz(output, payload) is True
        assert sha256_file(output) == original_hash
        changed = dict(payload, captions=np.asarray(['one', 'changed']))
        try:
            validate_or_write_npz(output, changed)
        except ValueError as error:
            assert 'refusing overwrite' in str(error)
        else:
            raise AssertionError('changed frozen request archive was overwritten')


if __name__ == '__main__':
    tests = (
        test_strict_cache_audit_accepts_exact_frozen_request_binding,
        test_strict_cache_audit_rejects_missing_per_sample_request_hash,
        test_frozen_request_writer_checks_indexed_name_and_caption,
        test_frozen_request_writer_preserves_official_and_request_counts,
        test_frozen_request_extractor_and_auditor_keep_distinct_counts,
        test_resume_wrapper_preserves_all_failed_attempt_sidecars,
        test_llm_robustness_metrics_distinguish_role_order_and_timing,
        test_rule_contact_parser_is_deterministic_and_fail_closed,
        test_frozen_request_invocation_records_success_and_failure_time,
        test_robustness_request_archive_is_byte_stable_and_fail_closed,
    )
    for test in tests:
        test()
        print({'test': test.__name__, 'status': 'pass'})
