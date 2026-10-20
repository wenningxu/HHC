#!/usr/bin/env python3

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_repredict_audit_is_parseable_and_content_bound():
    source = (ROOT / "tools/audit_hhc_native_repredict_semantics.py").read_text()
    ast.parse(source)
    assert "1770947689489_set-424054_change-424053_ContentChange.py" in source
    assert "source_immediately_precedes_low_fid_launch" in source
    assert "native_repredicts_all_valid_positions" in source
    assert '"promotion_policy": "requires frozen 544-sample validation comparison"' in source


if __name__ == "__main__":
    test_native_repredict_audit_is_parseable_and_content_bound()
