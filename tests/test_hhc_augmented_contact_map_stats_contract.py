from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_augmented_historical_root_cache_uses_root_stats():
    source = (ROOT / "tools/export_hhc_official_interx.py").read_text(
        encoding="utf-8"
    )
    assert "root_normalized_cache_contracts" in source
    assert "'historical_mmm_exact'" in source
    assert "'deterministic_historical_root_v1'" in source
    assert "cache_suffix = (" in source
    assert "in root_normalized_cache_contracts" in source


if __name__ == "__main__":
    test_augmented_historical_root_cache_uses_root_stats()
    print("augmented contact-map root-stat contract passed")
