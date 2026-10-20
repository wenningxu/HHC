from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT / path).read_text(encoding='utf-8')


def test_builder_materializes_same_random_sources_as_strict_control():
    text = source('tools/run_humanml3d_amass_shape_augmentation.sh')
    assert (
        'materialize_overlay random_shape_only "${random_overlay}"'
        in text
    )
    assert '"${full_random_positive}" "${full_random_zero}"' in text
    assert 'dataset/interx_shape_aug_amass_random_only_seed123_v1' in text
    assert "assert d.get('augmentation_mode') == sys.argv[3]" in text
    assert 'assert d[\'counts\'][\'augmented\'] == 6238' in text


def test_completed_materialization_is_validated_instead_of_rewritten():
    text = source('tools/run_humanml3d_amass_shape_augmentation.sh')
    assert 'if [[ -s "${manifest}" ]]' in text
    assert 'validate_materialized_interx_shape_augmentation' in text
    assert 'verify_augmented_motion_content=False' in text


def test_completed_pilot_gate_is_hash_validated_and_reused():
    text = source('tools/run_humanml3d_amass_shape_augmentation.sh')
    assert 'validated_gate_status()' in text
    assert "'gate_implementation' in artifacts" in text
    assert "assert digest(path) == row['sha256']" in text
    assert 'pilot_status="$(validated_gate_status "${pilot_gate}" || true)"' in text
    assert 'nearest_status="$(validated_gate_status "${pilot_nearest_gate}" || true)"' in text


def test_training_arms_share_one_launcher_and_online_wandb():
    builder = source('tools/run_humanml3d_amass_shape_augmentation.sh')
    launcher = source('tools/run_hhc_no_shape_amass_aug_pilot.sh')
    assert (
        'run_hhc_no_shape_amass_aug_pilot.sh "${gpu0}" optimized_selective'
        in builder
    )
    assert (
        'run_hhc_no_shape_amass_aug_pilot.sh "${gpu0}" random_only'
        in builder
    )
    assert 'optimized_selective)' in launcher
    assert 'random_only)' in launcher
    assert '--wandb-mode online' in launcher
    assert '--sampling-steps 50' in launcher
    assert '--seed 114514' in launcher


def test_finalizer_reports_all_three_paired_comparisons():
    matrix = source('tools/finalize_amass_shape_augmentation_matrix.sh')
    arm_finalizer = source('tools/finalize_no_shape_amass_aug_joint.sh')
    assert 'optimized_selective' in matrix
    assert 'random_only' in matrix
    assert 'random_vs_optimized_paired.json' in matrix
    assert '--replications 20' in matrix
    assert 'optimized_shape_augmentation' in arm_finalizer
    assert 'random_shape_augmentation' in arm_finalizer
