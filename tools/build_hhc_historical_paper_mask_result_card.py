#!/usr/bin/env python
"""Build a fail-closed result card for the completed HHC mask-ratio run."""

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output/t2i_trans/2026-09-02-11-49-08_hhc_historical_plus_paper_mask_ratio_raw12288_300k_seed114514_v2"
FILES = {
    "single_factor_contract": ROOT / "output/hhc_historical_plus_paper_mask_ratio_v2_single_factor_contract_audit.json",
    "runtime_mask_history": ROOT / "output/hhc_historical_plus_paper_mask_ratio_runtime_history_final_audit.json",
    "model_structure": ROOT / "output/hhc_historical_plus_paper_mask_ratio_model_structure_audit.json",
    "terminal_checkpoint": ROOT / "output/hhc_historical_plus_paper_mask_ratio_final_checkpoint_audit.json",
    "wandb_history": ROOT / "output/hhc_historical_plus_paper_mask_ratio_wandb_history_audit.json",
    "checkpoint_selection": ROOT / "output/hhc_historical_plus_paper_mask_ratio_selection.json",
    "sampling_runtime": ROOT / "output/hhc_sampling_runtime_audit.json",
    "interx_protocol": ROOT / "output/interx_protocol_audit.json",
    "no_timing_summary": ROOT / "output/fair_eval/hhc_historical_plus_paper_mask_ratio_no_contact_common50/summary.json",
    "oracle_timing_summary": ROOT / "output/fair_eval/hhc_historical_plus_paper_mask_ratio_contact_common50/summary.json",
    "training_manifest": RUN / "training_manifest.json",
}
METRICS = ("fid", "matching", "top1", "top2", "top3", "diversity", "multimodality")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(label):
    path = FILES[label]
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact_summary(payload, label):
    require(payload.get("replications") == 20, f"{label}: not 20 replications")
    require(payload.get("samples_per_replication") == 1696, f"{label}: not 1696 samples")
    result = {}
    for metric in METRICS:
        record = payload.get("metrics", {}).get(metric, {})
        values = record.get("values")
        require(isinstance(values, list) and len(values) == 20, f"{label}/{metric}: incomplete")
        require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in values), f"{label}/{metric}: non-finite")
        result[metric] = {"mean": float(record["mean"]), "ci95": float(record["ci95"])}
    return result


def main():
    single = load("single_factor_contract")
    mask = load("runtime_mask_history")
    structure = load("model_structure")
    checkpoint = load("terminal_checkpoint")
    wandb = load("wandb_history")
    selection = load("checkpoint_selection")
    sampling = load("sampling_runtime")
    protocol = load("interx_protocol")
    manifest = load("training_manifest")
    require(single.get("status") == "pass" and single["checks"].get("only_numerical_training_change_is_mask_contract") is True, "single-factor mask contract failed")
    require(mask.get("status") == "pass" and mask.get("through_optimizer_step") == 300000, "runtime mask history is incomplete")
    require(structure.get("status") == "pass" and structure["checks"].get("checkpoint_tensor_shapes_match") is True, "historical topology audit failed")
    require(checkpoint.get("status") == "pass" and checkpoint.get("global_step") == 300000, "terminal checkpoint audit failed")
    require(wandb.get("status") == "pass" and wandb.get("remote_state") == "finished", "online W&B audit failed")
    require(selection.get("status") == "complete" and selection.get("test_metrics_used_for_selection") is False, "validation-only checkpoint selection failed")
    require(selection.get("selected_step") == 260000, "selected checkpoint changed")
    require(sampling.get("status") == "pass" and sampling["checks"].get("full_length_executes_50_outer_iterations") is True, "50-step sampling audit failed")
    require(protocol.get("status") == "pass", "Inter-X protocol audit failed")
    args = manifest["resolved_arguments"]
    require(args.get("training_mask_contract") == "historical_pipeline_paper_kinematic_ratio", "training mask contract changed")
    require(args.get("training_loss_contract") == "historical_all_valid", "historical loss contract changed")
    require(args.get("paper_architecture") is False, "historical topology flag changed")

    no_timing = compact_summary(load("no_timing_summary"), "no timing")
    oracle = compact_summary(load("oracle_timing_summary"), "oracle timing")
    selected_path = Path(selection["selected_checkpoint"])
    require(sha256(selected_path) == selection["selected_checkpoint_sha256"], "selected checkpoint hash changed")
    payload = {
        "schema": "hhc_historical_plus_paper_mask_ratio_result_card_v1",
        "status": "complete",
        "claim": "performance-aligned single-factor reproduction: verified historical HHC training pipeline with only the paper asymmetric kinematic mask-ratio schedule substituted",
        "not_claimed": [
            "not the earlier multi-change 512D/8-head paper-configuration run",
            "not proof that the paper table checkpoint used this exact uncommitted historical source",
            "oracle contact timing is an ablation and not a deployable primary method",
        ],
        "training": {
            "optimizer_steps": 300000,
            "architecture": (
                "historical 1024D, 16 heads: a 9-block base composed of "
                "7 global self-attention plus 2 local cross-attention blocks, "
                "followed by 9 output self-attention blocks"
            ),
            "vq": "pre-cleanup raw 12288-code checkpoint at step 175000",
            "mask": "paper asymmetric Root/Body/Hands cosine kinematic ratios",
            "loss": "historical all-valid-token NLL retained due paper prose/equation ambiguity",
            "optimizer": "historical AdamW betas (0.5,0.9), lr 2e-4, MultiStepLR",
            "wandb": {"entity": wandb["entity"], "project": wandb["project"], "run_id": wandb["run_id"], "status": wandb["remote_state"]},
        },
        "selection": {
            "split": "official Inter-X validation only",
            "selected_step": selection["selected_step"],
            "selected_validation_fid": selection["selected_validation_fid"],
            "test_metrics_used": False,
            "checkpoint": artifact(selected_path),
        },
        "evaluation": {
            "dataset": "official Inter-X test split",
            "replications": 20,
            "samples_per_replication": 1696,
            "sampler": (
                "sample_intra_cas Root->Body->Hands cascade with immediate "
                "within-iteration updates, 50 outer MaskGIT steps, fixed tokens "
                "never remasked, temperature 0.5 and text CFG 3.5"
            ),
            "no_contact_timing": no_timing,
            "oracle_contact_timing": oracle,
        },
        "evidence": {label: artifact(path) for label, path in FILES.items()},
        "implementation": artifact(Path(__file__)),
    }
    output = ROOT / "output/hhc_historical_paper_mask_result_card.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    markdown = output.with_suffix(".md")
    markdown.write_text(
        "# HHC historical + paper mask-ratio result card\n\n"
        "Status: `complete`\n\n"
        f"- Selected on official validation only: step {selection['selected_step']:,}, FID {selection['selected_validation_fid']:.4f}.\n"
        f"- Test, no contact timing: FID {no_timing['fid']['mean']:.4f} ± {no_timing['fid']['ci95']:.4f}.\n"
        f"- Test, oracle timing ablation: FID {oracle['fid']['mean']:.4f} ± {oracle['fid']['ci95']:.4f}.\n"
        f"- W&B: `{wandb['entity']}/{wandb['project']}/{wandb['run_id']}`, finished and metric-complete.\n\n"
        "This is a performance-aligned single-factor reproduction: the verified historical topology/loss/optimizer/cache are retained, and only the paper asymmetric kinematic mask-ratio rule is substituted.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output": str(output), "fid": no_timing["fid"]}, sort_keys=True))


if __name__ == "__main__":
    main()
