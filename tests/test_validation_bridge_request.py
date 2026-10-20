import hashlib
import json

import numpy as np

from tools.prepare_hhc_validation_bridge_requests import (
    PER_SAMPLE_KEYS,
    build_exact_order_payload,
)


def records_hash(records):
    return hashlib.sha256(json.dumps(
        records, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")).hexdigest()


def fixture():
    # Shared request is evaluator-sorted: b,a,d,c.  Training sampled a,b,c,d.
    names = np.asarray(["b", "a", "d", "c"])
    captions = np.asarray(["B", "A", "D", "C"])
    lengths = np.asarray([8, 4, 16, 12], dtype=np.int64)
    source = {
        "names": names, "captions": captions, "lengths": lengths,
        "text_embeddings": np.arange(8).reshape(4, 2),
        "gt_embeddings": np.arange(12).reshape(4, 3),
        "betas": np.arange(16).reshape(4, 4),
        "gender": np.arange(8).reshape(4, 2),
        "contact": np.arange(20).reshape(4, 5),
    }
    records = [
        {"batch": 0, "name": "a", "caption": "A", "length": 4},
        {"batch": 0, "name": "b", "caption": "B", "length": 8},
        {"batch": 1, "name": "c", "caption": "C", "length": 12},
        {"batch": 1, "name": "d", "caption": "D", "length": 16},
    ]
    selection = {
        "records": records, "evaluated_samples": 4, "batch_size": 2,
        "seed": 500123, "ordered_request_records_sha256": records_hash(records),
    }
    return source, selection


def test_bridge_restores_training_order_without_changing_sample_fields():
    source, selection = fixture()
    output, indices = build_exact_order_payload(source, selection)
    assert indices.tolist() == [1, 0, 3, 2]
    assert output["names"].tolist() == ["a", "b", "c", "d"]
    for key in PER_SAMPLE_KEYS:
        np.testing.assert_array_equal(output[key], source[key][indices])
    assert output["source_common_request_indices"].tolist() == indices.tolist()
    assert output["seed"].item() == 500123


def test_bridge_rejects_caption_mismatch():
    source, selection = fixture()
    selection["records"][0]["caption"] = "wrong"
    selection["ordered_request_records_sha256"] = records_hash(selection["records"])
    try:
        build_exact_order_payload(source, selection)
    except ValueError as error:
        assert "fields differ" in str(error)
    else:
        raise AssertionError("caption mismatch was accepted")


if __name__ == "__main__":
    test_bridge_restores_training_order_without_changing_sample_fields()
    test_bridge_rejects_caption_mismatch()
