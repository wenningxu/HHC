#!/usr/bin/env python3
"""Fail closed unless the new HHC run differs causally only in mask ratio."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


DEFAULT_REFERENCE = Path(
    "output/t2i_trans/2026-08-29-16-44-04_"
    "mgt_known_good_historical_contract_v7_trainmode_runtime_loss_sampler_"
    "seed114514/training_manifest.json"
)


def launcher_command(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"#.*", "", source)
    source = source.replace("\\\n", " ")
    start = source.index("python -u train/train_t2i_trans.py")
    command = shlex.split(source[start:])
    substitutions = {
        "${experiment}": "candidate",
        "${vq_checkpoint}": "checkpoints/vq_precleanup_raw12288/net_175000.pth",
        "${cache_root}": "checkpoints/vq_precleanup_raw12288_historical_mmm_cache",
        "${cache_root}/codebook": (
            "checkpoints/vq_precleanup_raw12288_historical_mmm_cache/codebook"
        ),
    }
    return [substitutions.get(token, token) for token in command]


def option_map(command: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    index = next(i for i, token in enumerate(command) if token.startswith("--"))
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected command token: {token}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            result[token] = command[index + 1]
            index += 2
        else:
            result[token] = True
            index += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))["command"]
    expected = option_map(reference)
    actual = option_map(launcher_command(args.launcher))
    allowed_observability = {
        "--exp-name", "--wandb-mode", "--save-all-validation-checkpoints"
    }
    differences = {
        key: {"reference": expected.get(key), "candidate": actual.get(key)}
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    }
    required_differences = {
        "--training-mask-contract": {
            "reference": "historical_flat_random_replace",
            "candidate": "historical_pipeline_paper_kinematic_ratio",
        },
        "--wandb-mode": {"reference": "offline", "candidate": "online"},
    }
    checks = {
        "required_mask_change_exact": differences.get(
            "--training-mask-contract"
        ) == required_differences["--training-mask-contract"],
        "wandb_changed_to_online": differences.get("--wandb-mode")
        == required_differences["--wandb-mode"],
        "no_other_causal_argument_changes": not (
            set(differences)
            - allowed_observability
            - {"--training-mask-contract"}
        ),
        "historical_model_topology": actual.get("--no-paper-architecture") is True,
        "historical_all_valid_loss": actual.get("--training-loss-contract")
        == "historical_all_valid",
        "historical_optimizer": actual.get("--optimizer-contract")
        == "historical_multistep",
        "historical_token_window": actual.get(
            "--training-token-length-contract"
        ) == "historical_38",
        "historical_rng_trajectory": actual.get(
            "--training-validation-contract"
        ) == "historical_test_side_effect",
        "all_validation_candidates_retained": actual.get(
            "--save-all-validation-checkpoints"
        ) is True,
    }
    payload = {"status": "pass" if all(checks.values()) else "fail",
               "checks": checks, "differences": differences}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
