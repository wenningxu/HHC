#!/usr/bin/env python
"""Runtime-audit HHC's cascaded MaskGIT step and compute semantics.

The public sampling contract counts outer MaskGIT iterations.  Root, body and
hands are attempted in that order while unresolved masks remain; therefore
three channel stages and six CFG forwards per outer iteration are maxima, not
an unconditional amount of work for very short sequences.
"""

import argparse
import ast
import hashlib
import inspect
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import models.two_person_transformer as transformer_module
from models.two_person_transformer import Text2Motion_Transformer

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TraceModel:
    num_vq = 8
    block_size = 51

    def __init__(self, paper_architecture):
        self.paper_architecture = paper_architecture
        self.forward_calls = []

    def forward_function(self, ids, clip_feature, src_token_mask, **kwargs):
        del clip_feature, src_token_mask
        self.forward_calls.append({
            "force_mask": bool(kwargs.get("force_mask", False)),
            "masked_per_channel": [
                int((ids[:, :, channel] == self.num_vq + 2).sum().item())
                for channel in range(3)
            ],
        })
        batch, tokens, channels = ids.shape
        logits = torch.zeros(batch, tokens, channels, self.num_vq)
        # Avoid ties while keeping the audit deterministic at temperature 0.
        logits[..., 0] = 2.0
        logits[..., 1] = 1.0
        return logits


def run_case(motion_length, max_steps=50, *, paper_architecture=False):
    model = TraceModel(paper_architecture=paper_architecture)
    observed_outer_iterations = []
    original_schedule = transformer_module.cosine_schedule

    def traced_schedule(value):
        observed_outer_iterations.append(float(value.item()))
        return original_schedule(value)

    transformer_module.cosine_schedule = traced_schedule
    try:
        ids = Text2Motion_Transformer.sample_intra_cas(
            model,
            torch.zeros(1, 16),
            torch.zeros(1, 1, 16),
            m_length=torch.tensor([motion_length]),
            rand_pos=False,
            CFG=3.5,
            max_steps=max_steps,
            temperature=0.5,
        )
    finally:
        transformer_module.cosine_schedule = original_schedule

    conditional = [call for call in model.forward_calls if not call["force_mask"]]
    unconditional = [call for call in model.forward_calls if call["force_mask"]]
    token_length = (motion_length + 3) // 4
    end_id = model.num_vq
    pad_id = model.num_vq + 1
    return {
        "architecture": "paper_no_end" if paper_architecture else "historical_end_token",
        "motion_length_frames": motion_length,
        "motion_length_tokens": token_length,
        "requested_outer_iterations": max_steps,
        "observed_outer_iterations": len(observed_outer_iterations),
        "last_schedule_fraction": observed_outer_iterations[-1],
        "active_channel_stages": len(conditional),
        "conditional_forward_calls": len(conditional),
        "unconditional_forward_calls": len(unconditional),
        "total_forward_calls_with_cfg": len(model.forward_calls),
        "first_outer_conditional_mask_counts_body_hand_root": [
            call["masked_per_channel"] for call in conditional[:3]
        ],
        "unresolved_valid_masks_at_return": int(
            (ids == model.num_vq + 2).sum().item()
        ),
        "end_markers_at_expected_position": int(
            (ids[:, token_length, :] == end_id).sum().item()
        ) if token_length < ids.shape[1] else 0,
        "nonpadding_values_after_end": int(
            (ids[:, token_length + 1:, :] != pad_id).sum().item()
        ) if token_length + 1 < ids.shape[1] else 0,
    }


def range_contract_from_ast():
    source = inspect.getsource(Text2Motion_Transformer.sample_intra_cas)
    tree = ast.parse(textwrap.dedent(source))
    ranges = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.comprehension)) or not isinstance(node.iter, ast.Call):
            continue
        if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            ranges.append(ast.unparse(node.iter))
    return source, ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=ROOT / "output/hhc_sampling_runtime_audit.json",
        type=Path,
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    # 150 frames is the maximum official Inter-X evaluation length (38 VQ
    # tokens after ceil(length / 4)), so this is the relevant maximum-work
    # case rather than an out-of-protocol synthetic 200-frame sequence.
    # The promoted model intentionally retains the strong historical 1024-D
    # topology, including explicit end tokens.  Audit that deployed branch as
    # the authority; retain the paper no-end branch as a parity diagnostic.
    full = run_case(150, paper_architecture=False)
    short = run_case(4, paper_architecture=False)
    paper_full = run_case(150, paper_architecture=True)
    method_source, ranges = range_contract_from_ast()

    checks = {
        "source_outer_loop_is_one_through_max_steps": (
            "range(1, max_steps + 1)" in ranges
        ),
        "full_length_executes_50_outer_iterations": (
            full["observed_outer_iterations"] == 50
            and full["last_schedule_fraction"] == 1.0
        ),
        "full_length_executes_150_active_channel_stages": (
            full["active_channel_stages"] == 150
        ),
        "full_length_executes_300_cfg_forwards": (
            full["conditional_forward_calls"] == 150
            and full["unconditional_forward_calls"] == 150
            and full["total_forward_calls_with_cfg"] == 300
        ),
        "cascade_order_is_root_body_hands": (
            full["first_outer_conditional_mask_counts_body_hand_root"]
            == [[38, 38, 38], [38, 38, 37], [37, 38, 37]]
        ),
        "short_length_still_executes_50_outer_iterations": (
            short["observed_outer_iterations"] == 50
            and short["last_schedule_fraction"] == 1.0
        ),
        "short_length_skips_resolved_channel_forwards": (
            short["active_channel_stages"] == 3
            and short["total_forward_calls_with_cfg"] == 6
        ),
        "all_valid_masks_resolved": (
            full["unresolved_valid_masks_at_return"] == 0
            and short["unresolved_valid_masks_at_return"] == 0
        ),
        "historical_architecture_end_and_padding_are_preserved": (
            full["end_markers_at_expected_position"] == 3
            and short["end_markers_at_expected_position"] == 3
            and full["nonpadding_values_after_end"] == 0
            and short["nonpadding_values_after_end"] == 0
        ),
        "paper_no_end_branch_has_identical_outer_cascade_budget": (
            paper_full["observed_outer_iterations"] == 50
            and paper_full["active_channel_stages"] == 150
            and paper_full["total_forward_calls_with_cfg"] == 300
            and paper_full["end_markers_at_expected_position"] == 0
        ),
    }
    result = {
        "schema": "hhc_sampling_runtime_audit_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "contract": {
            "sampler": "sample_intra_cas",
            "model_architecture_contract": "historical_1024d_with_end_token",
            "sampling_steps": 50,
            "sampling_steps_unit": "outer_maskgit_iterations",
            "cascade_order": ["root", "body", "hands"],
            "maximum_active_channel_stages_per_outer_iteration": 3,
            "maximum_model_forward_calls_per_outer_iteration_with_cfg": 6,
            "maximum_common50_active_channel_stages": 150,
            "maximum_common50_model_forward_calls_with_cfg": 300,
            "resolved_channels_skip_forward_but_not_outer_iterations": True,
        },
        "full_length_runtime": full,
        "one_token_runtime": short,
        "paper_no_end_diagnostic_runtime": paper_full,
        "source_ranges": ranges,
        "checks": checks,
        "source": {
            "path": str(Path(inspect.getsourcefile(Text2Motion_Transformer)).resolve()),
            "sha256": sha256_file(inspect.getsourcefile(Text2Motion_Transformer)),
            "sample_intra_cas_source_sha256": hashlib.sha256(
                method_source.encode("utf-8")
            ).hexdigest(),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "interpretation": (
            "Fifty steps means fifty outer MaskGIT schedule iterations. The "
            "three cascade stages and six CFG forwards are exact for a "
            "full-length unresolved sequence and upper bounds in general."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
