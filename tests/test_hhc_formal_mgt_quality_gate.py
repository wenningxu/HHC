#!/usr/bin/env python3

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.assess_hhc_formal_mgt_quality import assess


def inputs():
    candidate = json.loads((
        ROOT / "output/fair_eval/hhc_formal_raw12288_repaired_mgt_common50_contact/replication_00_current_scorer_20260901.json"
    ).read_text())
    reference = json.loads((
        ROOT / "output/fair_eval/hhc_mgt_repair_v7_step280k_common50_paper_text_cfg_contact_rep00/replication_00_current_scorer_20260901.json"
    ).read_text())
    selection = json.loads((
        ROOT / "output/hhc_formal_raw12288_repaired_mgt_selection.json"
    ).read_text())
    contract = json.loads((
        ROOT / "output/hhc_formal_raw12288_repaired_mgt_training_contract_audit.json"
    ).read_text())
    policy = json.loads((
        ROOT / "configs/hhc_formal_mgt_quality_gate_v1.json"
    ).read_text())
    return candidate, reference, selection, contract, policy


def test_current_formal_model_is_rejected_on_quality_not_protocol():
    candidate, reference, selection, contract, policy = inputs()
    report, passed = assess(candidate, reference, selection, contract, policy)
    assert passed is False
    assert all(report["candidate_protocol_checks"].values())
    assert all(report["reference_protocol_checks"].values())
    assert all(report["training_contract_checks"].values())
    assert report["quality_checks"]["candidate_fid_at_or_below_limit"] is False
    assert report["quality_checks"]["candidate_minus_reference_at_or_below_limit"] is False
    assert report["observed"]["candidate_fid"] > 0.46


def test_protocol_identical_low_fid_candidate_passes_operational_gate():
    candidate, reference, selection, contract, policy = inputs()
    candidate = copy.deepcopy(candidate)
    candidate["fid"] = 0.25
    report, passed = assess(candidate, reference, selection, contract, policy)
    assert passed is True
    assert all(report["quality_checks"].values())


if __name__ == "__main__":
    test_current_formal_model_is_rejected_on_quality_not_protocol()
    test_protocol_identical_low_fid_candidate_passes_operational_gate()
