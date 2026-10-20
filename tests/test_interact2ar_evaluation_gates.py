import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import score_interact2ar_joint_evaluators as scorer  # noqa: E402


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_author_audit(paper_artifact):
    return {
        "schema": "interact2ar_author_release_calibration_audit_v1",
        "status": "pass",
        "checks": {"author_protocol": True, "paper_mean": True},
        "current_paper_repository_data_audit": paper_artifact,
    }


class Interact2AREvaluationGateTest(unittest.TestCase):
    def test_joint_scorer_accepts_current_passing_author_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper = root / "paper_repo_data.json"
            write_json(paper, {"status": "pass"})
            paper_artifact = scorer.artifact(paper)
            author = root / "author_calibration.json"
            write_json(author, valid_author_audit(paper_artifact))

            observed = scorer.load_author_calibration_audit(
                author, paper_artifact
            )

            self.assertEqual(observed, scorer.artifact(author))

    def test_joint_scorer_rejects_invalid_author_calibration(self):
        for mutation in ("failed_status", "failed_check", "stale_paper"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paper = root / "paper_repo_data.json"
                write_json(paper, {"status": "pass"})
                paper_artifact = scorer.artifact(paper)
                payload = valid_author_audit(paper_artifact)
                if mutation == "failed_status":
                    payload["status"] = "fail"
                elif mutation == "failed_check":
                    payload["checks"]["paper_mean"] = False
                else:
                    payload["current_paper_repository_data_audit"] = {
                        **paper_artifact,
                        "sha256": "0" * 64,
                    }
                author = root / "author_calibration.json"
                write_json(author, payload)

                with self.assertRaisesRegex(
                    ValueError, "author-release calibration"
                ):
                    scorer.load_author_calibration_audit(
                        author, paper_artifact
                    )


if __name__ == "__main__":
    unittest.main()
