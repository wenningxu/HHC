import copy

from tools.aggregate_fair_interhuman_metrics import summarize_records


IDENTITY = {
    'checkpoint_sha256': 'mgt',
    'vq_checkpoint_sha256': 'vq',
    'model_config_sha256': 'config',
    'generation_exporter_sha256': 'exporter',
    'interhuman_generation_bridge_sha256': 'bridge',
    'sampler': 'sample_intra_cas',
    'sampling_steps': 50,
    'temperature': 0.5,
    'cfg_scale': 3.5,
    'mgt_shape_condition': False,
    'contact_conditioning': False,
}
EVALUATOR = {'checkpoint': {'sha256': 'evaluator'}}


def _fixtures():
    main, mm, main_entries, mm_entries = [], [], [], []
    for replication in range(20):
        main_entry = {
            'replication': replication,
            'sha256': f'main-file-{replication}',
            'request_sha256': f'main-content-{replication}',
        }
        mm_entry = {
            'replication': replication,
            'sha256': f'mm-file-{replication}',
            'request_sha256': f'mm-content-{replication}',
        }
        main_entries.append(main_entry)
        mm_entries.append(mm_entry)
        main.append({
            'schema': 'interhuman_fair_main_score_v1',
            'status': 'complete',
            'replication': replication,
            'samples': 1056,
            'fid': replication / 10,
            'matching': 3.0,
            'top1': 0.4,
            'top2': 0.6,
            'top3': 0.7,
            'diversity': 8.0,
            'diversity_gt': 8.2,
            'request_file_sha256': main_entry['sha256'],
            'request_content_sha256': main_entry['request_sha256'],
            'generation_metadata': copy.deepcopy(IDENTITY),
            'evaluator_provenance': copy.deepcopy(EVALUATOR),
        })
        mm.append({
            'schema': 'interhuman_fair_mm_score_v1',
            'status': 'complete',
            'replication': replication,
            'multimodality': 1.0 + replication / 100,
            'request_file_sha256': mm_entry['sha256'],
            'request_content_sha256': mm_entry['request_sha256'],
            'generation_metadata': copy.deepcopy(IDENTITY),
            'evaluator_provenance': copy.deepcopy(EVALUATOR),
        })
    return main, mm, {'replications': main_entries}, {'replications': mm_entries}


def test_aggregate_requires_and_summarizes_full_paired_protocol():
    main, mm, main_manifest, mm_manifest = _fixtures()
    result = summarize_records(
        main, mm, main_manifest, mm_manifest, verify_generation_hashes=False
    )
    assert result['replications'] == 20
    assert result['model_sampler_identity'] == IDENTITY
    assert abs(result['metrics']['fid']['mean'] - 0.95) < 1e-12
    assert abs(result['metrics']['multimodality']['mean'] - 1.095) < 1e-12


def test_aggregate_rejects_main_mm_checkpoint_or_sampler_drift():
    main, mm, main_manifest, mm_manifest = _fixtures()
    mm[7]['generation_metadata']['checkpoint_sha256'] = 'different'
    try:
        summarize_records(
            main, mm, main_manifest, mm_manifest, verify_generation_hashes=False
        )
    except ValueError as exc:
        assert 'main and MM model/sampler identities differ' in str(exc)
    else:
        raise AssertionError('main/MM checkpoint drift was accepted')


def test_aggregate_rejects_request_binding_or_missing_replication():
    main, mm, main_manifest, mm_manifest = _fixtures()
    main[3]['request_file_sha256'] = 'wrong'
    try:
        summarize_records(
            main, mm, main_manifest, mm_manifest, verify_generation_hashes=False
        )
    except ValueError as exc:
        assert 'main request_file_sha256 mismatch' in str(exc)
    else:
        raise AssertionError('stale main request binding was accepted')

    main, mm, main_manifest, mm_manifest = _fixtures()
    try:
        summarize_records(
            main[:-1], mm, main_manifest, mm_manifest, verify_generation_hashes=False
        )
    except ValueError as exc:
        assert '20 main and 20 MM' in str(exc)
    else:
        raise AssertionError('incomplete 20-rep aggregate was accepted')
