import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_post_frozen_handoff_readiness import canonical_sha256


def test_contract_hash_excludes_timestamp_and_self_hash():
    path = ROOT / (
        "output/interx_shape_aug_pilot_optimized_flathand_"
        "frozenfoot_query10475_seed20260828/_run_contract.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    canonical = {
        key: value for key, value in contract.items()
        if key not in ("created_at_utc", "contract_sha256")
    }
    assert canonical_sha256(canonical) == contract["contract_sha256"]


if __name__ == "__main__":
    test_contract_hash_excludes_timestamp_and_self_hash()
