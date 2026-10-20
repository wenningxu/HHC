import importlib.util
import io
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paired", ROOT / "tools/compare_interact2ar_joint_augmentation_paired.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CARD_SPEC = importlib.util.spec_from_file_location(
    "amass_card", ROOT / "tools/build_amass_shape_augmentation_result_card.py"
)
CARD = importlib.util.module_from_spec(CARD_SPEC)
CARD_SPEC.loader.exec_module(CARD)


def test_metric_directions_and_diversity_distance_to_gt():
    baseline = {
        "fid": 1.0, "matching": 3.0, "top1": 0.4, "top2": 0.6,
        "top3": 0.7, "diversity": 11.0, "diversity_gt": 9.0,
        "multimodality": 1.0,
    }
    augmented = {
        "fid": 0.8, "matching": 2.9, "top1": 0.5, "top2": 0.65,
        "top3": 0.72, "diversity": 9.5, "diversity_gt": 9.0,
        "multimodality": 1.2,
    }
    for metric in MODULE.METRICS:
        assert MODULE.metric_improvement(metric, baseline, augmented) > 0


def test_diversity_overshoot_is_not_automatically_better():
    baseline = {"diversity": 9.1, "diversity_gt": 9.0}
    augmented = {"diversity": 10.0, "diversity_gt": 9.0}
    assert MODULE.metric_improvement("diversity", baseline, augmented) < 0


def test_amass_three_arm_comparison_kinds_have_distinct_schemas_and_roles():
    kinds = (
        "random_shape_augmentation",
        "optimized_shape_augmentation",
        "optimization_effect_with_fixed_amass_donors",
    )
    schemas = [MODULE.COMPARISON_SCHEMAS[kind] for kind in kinds]
    roles = [MODULE.COMPARISON_CANDIDATE_ROLES[kind] for kind in kinds]
    assert len(set(schemas)) == len(kinds)
    assert len(set(roles)) == len(kinds)
    assert all(schema.endswith('_paired_v1') for schema in schemas)


def test_replication_set_artifact_binds_ordered_file_contents(tmp_path):
    for index, value in enumerate(('left', 'right')):
        (tmp_path / f'replication_{index:02d}.json').write_text(value)
    before = MODULE.replication_set_artifact(tmp_path, 2)
    (tmp_path / 'replication_01.json').write_text('changed')
    after = MODULE.replication_set_artifact(tmp_path, 2)
    assert before['replication_files'] == after['replication_files'] == 2
    assert before['ordered_name_size_sha256_digest'] != (
        after['ordered_name_size_sha256_digest']
    )
    (tmp_path / 'replication_02.json').write_text('extra')
    try:
        MODULE.replication_set_artifact(tmp_path, 2)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpected extra replication was accepted')


def test_fixed_donor_optimization_kind_runs_and_binds_evidence(tmp_path):
    baseline = tmp_path / 'baseline'
    candidate = tmp_path / 'candidate'
    baseline.mkdir()
    candidate.mkdir()
    for index in range(20):
        common = {
            'schema': 'interact2ar_joint_evaluator_replication_v1',
            'status': 'complete', 'replication': index, 'samples': 1696,
            'mm_samples': 100, 'mm_repeats': 30,
            'main_generation_metadata': {'requests_sha256': f'main-{index}'},
            'mm_generation_metadata': {'requests_sha256': f'mm-{index}'},
            'requests': {'sha256': f'request-{index}'},
            'raw_ground_truth': {'sha256': 'gt'},
            'evaluator_provenance': {'weights': 'same'},
        }
        base_metric = {
            'fid': 1.0 + index * 1e-3, 'matching': 3.0,
            'top1': 0.4, 'top2': 0.6, 'top3': 0.7,
            'diversity': 10.0, 'diversity_gt': 9.0,
            'multimodality': 1.0,
        }
        candidate_metric = {
            **base_metric, 'fid': base_metric['fid'] - 0.1,
            'matching': 2.9, 'top1': 0.45, 'top2': 0.65,
            'top3': 0.75, 'diversity': 9.5, 'multimodality': 1.2,
        }
        for root, metric in ((baseline, base_metric), (candidate, candidate_metric)):
            record = {**common, 'metrics': {
                wrapper: dict(metric) for wrapper in MODULE.WRAPPERS
            }}
            (root / f'replication_{index:02d}.json').write_text(json.dumps(record))
    output = tmp_path / 'paired.json'
    previous = sys.argv
    try:
        sys.argv = [
            'paired', '--baseline', str(baseline), '--augmented', str(candidate),
            '--comparison-kind', 'optimization_effect_with_fixed_amass_donors',
            '--claim-scope', 'matched_training', '--output', str(output),
        ]
        with redirect_stdout(io.StringIO()):
            MODULE.main()
    finally:
        sys.argv = previous
    result = json.loads(output.read_text())
    assert result['schema'] == (
        'hhc_interact2ar_joint_shape_optimization_effect_paired_v1'
    )
    assert result['candidate_role'] == (
        'optimized_candidate_against_same_donor_random_control'
    )
    assert result['primary_all_improve'] is True
    assert result['evidence']['baseline_replications'] == (
        MODULE.replication_set_artifact(baseline, 20)
    )
    assert list(tmp_path.glob('paired.json.tmp.*')) == []
    verified = CARD.verify_pair(
        output, 'optimization_effect_with_fixed_amass_donors',
        baseline, candidate,
    )
    assert verified['primary_all_improve'] is True
    changed = candidate / 'replication_00.json'
    changed.write_text(changed.read_text() + '\n')
    try:
        CARD.verify_pair(
            output, 'optimization_effect_with_fixed_amass_donors',
            baseline, candidate,
        )
    except ValueError:
        pass
    else:
        raise AssertionError('result card accepted stale paired evidence')
