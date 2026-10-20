from tools.compare_interhuman_vq_validation import compare


def _records():
    common = {
        'status': 'complete', 'split': 'val', 'samples': 480,
        'request_file_sha256': 'request-file',
        'request_content_sha256': 'request-content',
        'evaluator_provenance': {'checkpoint': 'same'},
        'fid': 0.2, 'matching': 3.0, 'mpjpe_m': 0.1,
        'normalized_feature_mae': 0.3, 'top1': 0.5, 'top2': 0.6,
        'top3': 0.7, 'diversity': 8.0, 'diversity_gt': 8.1,
    }
    hhc = dict(common, schema='hhc_interhuman_vq_official_validation_score_v1')
    official = dict(
        common, schema='interhuman_vq_official_validation_score_v1',
        fid=0.3, top1=0.4,
    )
    return hhc, official


def test_vq_comparison_uses_matched_val_and_metric_directions():
    hhc, official = _records()
    result = compare(hhc, official)
    assert result['same_request_and_evaluator']
    assert result['metrics']['fid']['hhc_better']
    assert result['metrics']['top1']['hhc_better']
    assert result['scope'].endswith('descriptive_not_test')


def test_vq_comparison_rejects_request_or_evaluator_drift():
    for key in ('request_file_sha256', 'evaluator_provenance'):
        hhc, official = _records()
        official[key] = 'different'
        try:
            compare(hhc, official)
        except ValueError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f'VQ comparison accepted {key} drift')


def test_diversity_is_closer_to_gt_not_larger_is_better():
    hhc, official = _records()
    hhc['diversity'] = 9.0
    result = compare(hhc, official)['metrics']['diversity']
    assert result['better_direction'] == 'closer_to_gt'
    assert not result['hhc_better']
    hhc['diversity'] = 8.1
    assert compare(hhc, official)['metrics']['diversity']['hhc_better']


def test_comparison_rejects_nonfinite_metrics_and_gt_drift():
    for name, value in (('fid', float('nan')), ('diversity', float('inf')),
                        ('diversity_gt', 9.1)):
        hhc, official = _records()
        hhc[name] = value
        try:
            compare(hhc, official)
        except ValueError as error:
            assert name in str(error)
        else:
            raise AssertionError(f'accepted invalid {name}')
