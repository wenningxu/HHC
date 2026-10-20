import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'amass_card', ROOT / 'tools/build_amass_shape_augmentation_result_card.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_training_normalization_allows_only_arm_asset_identity():
    random = {
        'exp_name': 'random', 'cache_root': '/random/cache',
        'augmentation_mode': 'random_shape_only',
        'train_asset_root': '/random/overlay',
        'training_mask_contract': 'historical_pipeline_paper_kinematic_ratio',
        'sampling_steps': 50, 'lr': 0.0002,
    }
    optimized = {
        **random, 'exp_name': 'optimized', 'cache_root': '/optimized/cache',
        'augmentation_mode': 'optimized_positive_plus_random_zero',
        'train_asset_root': '/optimized/overlay',
    }
    assert MODULE.normalized_training_args(random) == (
        MODULE.normalized_training_args(optimized)
    )
    optimized['lr'] = 0.0003
    assert MODULE.normalized_training_args(random) != (
        MODULE.normalized_training_args(optimized)
    )


def test_matrix_finalizer_invokes_fail_closed_result_card():
    source = (ROOT / 'tools/finalize_amass_shape_augmentation_matrix.sh').read_text()
    assert 'tools/build_amass_shape_augmentation_result_card.py' in source
    assert '--optimized-run "${optimized_run}"' in source
    assert '--random-run "${random_run}"' in source
