from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_hhc_paper_training_contract import (
    token_cache_contract_mismatches,
)


FIELDS = (
    'token_cache_contract', 'vq_checkpoint_sha256', 'nb_code', 'code_dim',
    'down_t', 'cache_seed', 'cache_crop_policy', 'cache_scope',
    'cache_input_filter', 'tokenizer_implementation_sha256',
    'normalization_mean_sha256', 'normalization_std_sha256',
    'code_cache_sha256', 'contact_cache_sha256', 'expected_samples',
    'cached_code_samples', 'cached_contact_samples',
    'eligible_train_samples', 'eligible_val_samples', 'contact_map_scope',
    'contact_map_estimator', 'contact_map_samples', 'contact_map_sha256',
    'train_split_sha256', 'val_split_sha256', 'status',
)


def fixtures():
    h = 'a' * 64
    canonical = {
        'token_cache_contract': 'historical_mmm_exact',
        'vq_checkpoint_sha256': 'b' * 64,
        'nb_code': 12288,
        'code_dim': 256,
        'down_t': 2,
        'normalization_mean_sha256': 'c' * 64,
        'normalization_std_sha256': 'd' * 64,
        'train_split_sha256': 'e' * 64,
        'val_split_sha256': 'f' * 64,
    }
    active = {
        **canonical,
        'token_cache_contract': 'deterministic_historical_root_v1',
        'normalization_mean_sha256': '2' * 64,
        'normalization_std_sha256': '3' * 64,
        'tokenizer_implementation_sha256': '4' * 64,
        'cache_seed': 123,
        'cache_crop_policy': 'source_single_single_double_per_sample_sha256_v1',
        'cache_scope': 'train+val',
        'cache_input_filter': 'missing_eligible_train_val_before_tokenization',
        'code_cache_sha256': h,
        'contact_cache_sha256': h,
        'contact_map_sha256': h,
        'train_split_sha256': '1' * 64,
        'expected_samples': 15902,
        'cached_code_samples': 15902,
        'cached_contact_samples': 15902,
        'eligible_train_samples': 15348,
        'eligible_val_samples': 554,
        'contact_map_scope': 'train_only',
        'contact_map_estimator': 'empirical_frequency_unseen_neutral_0.5',
        'contact_map_samples': 15348,
        'status': 'complete',
    }
    return canonical, active


def test_augmented_cache_accepts_changed_train_content_but_fixed_vq_stats_val():
    canonical, active = fixtures()
    assert token_cache_contract_mismatches(
        active, canonical, FIELDS, augmented=True,
        augmented_tokenizer_mean_sha256='2' * 64,
        augmented_tokenizer_std_sha256='3' * 64,
        augmented_tokenizer_implementation_sha256='4' * 64,
    ) == {}


def test_augmented_cache_rejects_unaugmented_train_and_incomplete_counts():
    canonical, active = fixtures()
    active['train_split_sha256'] = canonical['train_split_sha256']
    active['cached_code_samples'] -= 1
    mismatches = token_cache_contract_mismatches(
        active, canonical, FIELDS, augmented=True,
        augmented_tokenizer_mean_sha256='2' * 64,
        augmented_tokenizer_std_sha256='3' * 64,
        augmented_tokenizer_implementation_sha256='4' * 64,
    )
    assert 'train_split_differs_from_unaugmented' in mismatches
    assert 'cached_code_samples_equals_expected' in mismatches


def test_baseline_cache_still_requires_byte_exact_contract():
    canonical, active = fixtures()
    canonical.update({key: active.get(key) for key in FIELDS})
    canonical['token_cache_contract'] = 'historical_mmm_exact'
    active = dict(canonical)
    assert token_cache_contract_mismatches(
        active, canonical, FIELDS, augmented=False
    ) == {}
    active['code_cache_sha256'] = '0' * 64
    assert 'code_cache_sha256' in token_cache_contract_mismatches(
        active, canonical, FIELDS, augmented=False
    )


if __name__ == '__main__':
    test_augmented_cache_accepts_changed_train_content_but_fixed_vq_stats_val()
    test_augmented_cache_rejects_unaugmented_train_and_incomplete_counts()
    test_baseline_cache_still_requires_byte_exact_contract()
