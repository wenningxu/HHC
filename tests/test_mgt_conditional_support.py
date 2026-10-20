from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_mgt_train_val_conditional_support import (
    apply_training_loader_contract,
    token_window_contract,
)


def test_loader_contract_filters_before_truncating():
    records = [
        ("short", np.zeros((19, 3), dtype=np.int64)),
        ("minimum", np.zeros((20, 3), dtype=np.int64)),
        ("long", np.zeros((60, 3), dtype=np.int64)),
    ]
    active, excluded, truncated = apply_training_loader_contract(
        records, minimum_payload_tokens=20, maximum_payload_tokens=49
    )
    assert [name for name, _ in active] == ["minimum", "long"]
    assert [len(tokens) for _, tokens in active] == [20, 49]
    assert excluded == ["short"]
    assert truncated == 1


def test_manifest_token_windows_reserve_one_end_position():
    assert token_window_contract({
        "training_token_length_contract": "paper_50", "block_size": 51,
    }) == (50, 49, "paper_50")
    assert token_window_contract({
        "training_token_length_contract": "original_mmm_38", "block_size": 51,
    }) == (38, 37, "original_mmm_38")


if __name__ == "__main__":
    for test in (
        test_loader_contract_filters_before_truncating,
        test_manifest_token_windows_reserve_one_end_position,
    ):
        test()
        print({"test": test.__name__, "status": "pass"})
