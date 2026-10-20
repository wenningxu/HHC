import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pair_audit_requires_bitwise_state_and_metric_binding():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        full = root / "epoch=99-step=7100.ckpt"
        weights = root / "step_000007100_weights.ckpt"
        metrics = root / "step_000007100.json"
        output = root / "audit.json"
        state = {"model.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        torch.save({"global_step": 7100, "state_dict": state}, full)
        torch.save({"state_dict": state}, weights)
        metrics.write_text(json.dumps({
            "status": "complete",
            "samples": 544,
            "generation_request_hash_verified": True,
            "fid": 1.0,
            "diversity": 2.0,
            "matching": 3.0,
            "top1": 0.1,
            "top2": 0.2,
            "top3": 0.3,
            "main_generation_metadata": {
                "checkpoint": str(weights.resolve()),
                "checkpoint_sha256": sha256(weights),
            },
        }))
        command = [
            sys.executable,
            str(ROOT / "tools/audit_timotion_validation_checkpoint_pair.py"),
            "--full-checkpoint", str(full),
            "--weights-checkpoint", str(weights),
            "--metrics", str(metrics),
            "--expected-step", "7100",
            "--output", str(output),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        assert json.loads(output.read_text())["status"] == "pass"

        payload = torch.load(weights, weights_only=False)
        payload["state_dict"]["model.weight"][0, 0] += 1
        torch.save(payload, weights)
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert "tensor values differ" in result.stderr


if __name__ == "__main__":
    test_pair_audit_requires_bitwise_state_and_metric_binding()
    print("TiMotion validation checkpoint-pair audit test passed")
