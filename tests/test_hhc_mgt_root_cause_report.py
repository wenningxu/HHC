import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_cause_report_is_evidence_bound(tmp_path):
    output = tmp_path / "root_cause.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build_hhc_mgt_root_cause_report.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "diagnosis_narrowed_not_causally_isolated"
    assert all(payload["checks"].values())
    assert payload["quality_gap"]["current_formal_common_scorer_fid"] > 0.4
    assert payload["paper_table_linkage"]["absolute_difference"] <= 0.002
    assert (
        payload["ruled_out_as_unique_causes"]["token_update_support"]
        ["delta_repredict_minus_fixed"]
        > 0
    )
    assert len(payload["remaining_causal_set"]) >= 4


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_root_cause_report_is_evidence_bound(Path(directory))
