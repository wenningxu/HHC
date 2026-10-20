from tools.merge_fair_interx_shard_manifests import comparable


def test_comparable_ignores_only_range_and_completion_timestamp():
    base = {
        "method": "hhc",
        "replication_range": [0, 10],
        "checkpoint": "/checkpoint",
        "provenance": {"driver": "hash"},
        "implementation_source_stability_verified_at_completion": True,
        "implementation_source_stability_verified_at_utc": "first",
    }
    peer = dict(base)
    peer["replication_range"] = [10, 20]
    peer["implementation_source_stability_verified_at_utc"] = "second"
    assert comparable(base) == comparable(peer)
    peer["checkpoint"] = "/different"
    assert comparable(base) != comparable(peer)
