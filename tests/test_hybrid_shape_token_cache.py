import json
import tempfile
from pathlib import Path

import numpy as np

from utils.hybrid_shape_token_cache import (
    TOKEN_CACHE_CONTRACT,
    audit_hybrid_cache_subsets,
    import_original_pairs_byte_exact,
    partition_expected_names,
)


def save(path, values):
    np.save(path, np.asarray(values, dtype=np.int64))


def test_hybrid_cache_preserves_original_bytes_and_partitions_names():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "materialization.json"
        manifest.write_text(json.dumps({
            "schema": "hhc_interx_shape_augmentation_materialization_v1",
            "status": "complete",
            "asset_records": [{"augmented_name": "source__shapeaug_s1"}],
        }))
        original, augmented, _ = partition_expected_names(
            {"source", "val", "source__shapeaug_s1"}, manifest
        )
        assert original == {"source", "val"}
        assert augmented == {"source__shapeaug_s1"}

        historical_code = root / "historical_code"
        historical_contact = root / "historical_contact"
        cache_code = root / "cache_code"
        cache_contact = root / "cache_contact"
        historical_code.mkdir()
        historical_contact.mkdir()
        for index, name in enumerate(sorted(original)):
            save(historical_code / f"{name}.npy", [index, 10])
            save(historical_contact / f"{name}.npy", [index, 20])
        assert import_original_pairs_byte_exact(
            original, historical_code, historical_contact,
            cache_code, cache_contact,
        ) == 2
        # Resuming an identical import is allowed.
        assert import_original_pairs_byte_exact(
            original, historical_code, historical_contact,
            cache_code, cache_contact,
        ) == 2
        save(cache_code / "source__shapeaug_s1.npy", [30])
        save(cache_contact / "source__shapeaug_s1.npy", [40])
        audit = audit_hybrid_cache_subsets(
            original, augmented, historical_code, historical_contact,
            cache_code, cache_contact,
        )
        assert audit["status"] == "pass"
        assert audit["token_cache_contract"] == TOKEN_CACHE_CONTRACT
        assert audit["counts"] == {"original": 2, "augmented": 1, "total": 3}


def test_hybrid_cache_rejects_conflicting_original_and_scope_contamination():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        historical_code = root / "historical_code"
        historical_contact = root / "historical_contact"
        cache_code = root / "cache_code"
        cache_contact = root / "cache_contact"
        for path in (
            historical_code, historical_contact, cache_code, cache_contact,
        ):
            path.mkdir()
        save(historical_code / "original.npy", [1])
        save(historical_contact / "original.npy", [2])
        save(cache_code / "original.npy", [999])
        try:
            import_original_pairs_byte_exact(
                {"original"}, historical_code, historical_contact,
                cache_code, cache_contact,
            )
        except RuntimeError as error:
            assert "conflicts with MMM" in str(error)
        else:
            raise AssertionError("conflicting original cache was accepted")

        save(cache_code / "original.npy", [1])
        save(cache_contact / "original.npy", [2])
        save(cache_code / "augmented.npy", [3])
        save(cache_contact / "augmented.npy", [4])
        save(cache_code / "test_leak.npy", [5])
        try:
            audit_hybrid_cache_subsets(
                {"original"}, {"augmented"}, historical_code,
                historical_contact, cache_code, cache_contact,
            )
        except RuntimeError as error:
            assert "scope mismatch" in str(error)
            assert "test_leak" in str(error)
        else:
            raise AssertionError("hybrid cache accepted an out-of-scope token")


def test_hybrid_partition_rejects_malformed_or_out_of_scope_augmented_names():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "materialization.json"
        base = {
            "schema": "hhc_interx_shape_augmentation_materialization_v1",
            "status": "complete",
        }
        manifest.write_text(json.dumps({
            **base, "asset_records": [{"source_name": "missing_name"}],
        }))
        try:
            partition_expected_names({"original", "augmented"}, manifest)
        except ValueError as error:
            assert "invalid sample name" in str(error)
        else:
            raise AssertionError("missing augmented_name was accepted")

        manifest.write_text(json.dumps({
            **base, "asset_records": [{"augmented_name": "outside"}],
        }))
        try:
            partition_expected_names({"original", "augmented"}, manifest)
        except ValueError as error:
            assert "outside the eligible cache scope" in str(error)
        else:
            raise AssertionError("out-of-scope augmented name was accepted")


if __name__ == "__main__":
    test_hybrid_cache_preserves_original_bytes_and_partitions_names()
    test_hybrid_cache_rejects_conflicting_original_and_scope_contamination()
    test_hybrid_partition_rejects_malformed_or_out_of_scope_augmented_names()
    print("Hybrid shape token-cache tests passed")
