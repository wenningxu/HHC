#!/usr/bin/env python
"""Rebuild and checkpoint-validate the parameter counts used in the review.

Each repository is loaded in an isolated subprocess.  This avoids collisions
between the three projects' top-level ``models`` and ``utils`` packages and
also releases the large checkpoint/model allocations between methods.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HHC_EXPECTED = {
    "shape_aware_vq_parameters": 36_606_748,
    "mgt_parameters_excluding_shared_vq": 71_840_336,
    "unique_motion_parameters": 108_447_084,
    "frozen_clip_vit_b32_parameters": 151_277_313,
    "total_inference_resident_parameters": 259_724_397,
}
HHC_PRECLEANUP_RAW12288_EXPECTED = {
    "shape_aware_vq_parameters": 36_606_748,
    "mgt_parameters_excluding_shared_vq": 73_986_640,
    "unique_motion_parameters": 110_593_388,
    "frozen_clip_vit_b32_parameters": 151_277_313,
    "total_inference_resident_parameters": 261_870_701,
}
INTERMASK_EXPECTED = {
    "vq_parameters": 45_418_502,
    "transformer_total_parameters": 177_333_376,
    "transformer_trainable_parameters": 54_332_032,
    "unique_motion_parameters": 99_750_534,
    "total_inference_resident_parameters": 222_751_878,
}
COSHMDM_EXPECTED = {
    "total_parameters": 305_442_310,
    "trainable_parameters": 182_440_966,
    "frozen_parameters": 123_001_344,
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def source_artifacts(root, relative_paths):
    root = Path(root).resolve()
    return {
        relative: artifact(root / relative)
        for relative in relative_paths
    }


def count_unique(parameters):
    seen = set()
    total = 0
    for parameter in parameters:
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            total += parameter.numel()
    return total, seen


def require_expected(label, counts, expected):
    mismatches = {
        key: {"expected": value, "actual": counts.get(key)}
        for key, value in expected.items()
        if counts.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} parameter-count contract changed: {mismatches}")


def atomic_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def component_hhc(args):
    root = Path(args.hhc_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    import clip
    import torch

    from models.model_tools import build_model
    from tools.export_hhc_official_interx import paper_model_args
    from train.train_t2i_trans import (
        load_transformer_checkpoint,
        merge_interx_lookup_weights,
    )

    vq_checkpoint = Path(args.hhc_vq_checkpoint).resolve()
    mgt_checkpoint = Path(args.hhc_mgt_checkpoint).resolve()
    vq_payload = torch.load(
        vq_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    model_args = paper_model_args(
        "cpu", str(vq_checkpoint), vq_payload=vq_payload,
        contract=args.hhc_vq_contract,
    )
    vq = build_model(model_args, "interx_vq")
    vq_state = merge_interx_lookup_weights(dict(vq_payload["net"]))
    vq.load_state_dict(vq_state, strict=True)
    del vq_payload, vq_state

    transformer = build_model(model_args, "interx_transformer", vqvae=vq)
    load_transformer_checkpoint(
        transformer, str(mgt_checkpoint), preserve_vqvae=True
    )
    vq_count, vq_ids = count_unique(vq.parameters())
    transformer_count, transformer_ids = count_unique(transformer.parameters())
    if not vq_ids.issubset(transformer_ids):
        raise RuntimeError("HHC transformer does not retain the shared VQ instance")
    mgt_count = transformer_count - vq_count

    clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    clip_count, _ = count_unique(clip_model.parameters())
    counts = {
        "shape_aware_vq_parameters": vq_count,
        "mgt_parameters_excluding_shared_vq": mgt_count,
        "unique_motion_parameters": vq_count + mgt_count,
        "frozen_clip_vit_b32_parameters": clip_count,
        "total_inference_resident_parameters": (
            vq_count + mgt_count + clip_count
        ),
    }
    expected = (
        HHC_PRECLEANUP_RAW12288_EXPECTED
        if args.hhc_vq_contract == "precleanup_raw12288"
        else HHC_EXPECTED
    )
    require_expected("HHC", counts, expected)
    return {
        **counts,
        "vq_contract": args.hhc_vq_contract,
        "checkpoint_load": "VQ strict; MGT strict excluding the intentionally shared VQ subtree",
        "vq_checkpoint": artifact(vq_checkpoint),
        "mgt_checkpoint": artifact(mgt_checkpoint),
        "source_files": source_artifacts(root, [
            "models/model_tools.py",
            "models/vqvae_sep3_shape.py",
            "models/encdec.py",
            "models/quantize_cnn.py",
            "models/two_person_transformer.py",
            "train/train_t2i_trans.py",
            "tools/export_hhc_official_interx.py",
        ]),
        "torch_version": torch.__version__,
    }


def component_intermask(args):
    root = Path(args.intermask_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    import torch

    from tools_export_official_interx import load_models

    vq, transformer, transformer_checkpoint = load_models(
        torch.device("cpu"), str(root / "checkpoints"),
        "trans_default", "vq_default",
    )
    vq_count, _ = count_unique(vq.parameters())
    transformer_total, _ = count_unique(transformer.parameters())
    transformer_trainable, _ = count_unique(
        parameter for parameter in transformer.parameters()
        if parameter.requires_grad
    )
    counts = {
        "vq_parameters": vq_count,
        "transformer_total_parameters": transformer_total,
        "transformer_trainable_parameters": transformer_trainable,
        "unique_motion_parameters": vq_count + transformer_trainable,
        "total_inference_resident_parameters": vq_count + transformer_total,
    }
    require_expected("InterMask", counts, INTERMASK_EXPECTED)
    vq_checkpoint = root / "checkpoints/interx/vq_default/model/best_fid.tar"
    return {
        **counts,
        "checkpoint_load": "author VQ and transformer checkpoints validated by the fair exporter",
        "vq_checkpoint": artifact(vq_checkpoint),
        "transformer_checkpoint": artifact(transformer_checkpoint),
        "source_files": source_artifacts(root, [
            "models/vq/model.py",
            "models/vq/residual_vq.py",
            "models/mask_transformer/transformer.py",
            "models/mask_transformer/transformer_modules.py",
            "checkpoints/interx/vq_default/opt.txt",
            "checkpoints/interx/trans_default/opt.txt",
            "tools_export_official_interx.py",
        ]),
        "torch_version": torch.__version__,
    }


def component_coshmdm(args):
    root = Path(args.coshmdm_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    import torch

    from configs import get_config
    from models import CoShMDM
    from tools.export_official_interx import load_checkpoint

    cfg = get_config(str(root / "configs/model.yaml"))
    model = CoShMDM(cfg)
    checkpoint = Path(args.coshmdm_checkpoint).resolve()
    load_checkpoint(model, str(checkpoint))
    total, _ = count_unique(model.parameters())
    trainable, _ = count_unique(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = total - trainable
    counts = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }
    require_expected("CoShMDM", counts, COSHMDM_EXPECTED)
    return {
        **counts,
        "checkpoint_load": "strict fair-export checkpoint load",
        "checkpoint": artifact(checkpoint),
        "source_files": source_artifacts(root, [
            "configs/model.yaml",
            "models/coshmdm.py",
            "models/gaussian_diffusion.py",
            "models/nets.py",
            "models/blocks.py",
            "tools/export_official_interx.py",
        ]),
        "torch_version": torch.__version__,
    }


def component_main(args):
    functions = {
        "hhc": component_hhc,
        "intermask": component_intermask,
        "coshmdm": component_coshmdm,
    }
    result = functions[args.component](args)
    atomic_json(args.component_output, result)


def resolve_hhc_checkpoint(args):
    if args.hhc_selection_manifest:
        selection_path = Path(args.hhc_selection_manifest).resolve()
        with selection_path.open(encoding="utf-8") as handle:
            selection = json.load(handle)
        if not selection.get("run_complete"):
            raise ValueError("HHC selection manifest does not prove a complete run")
        selected = Path(selection["selected_checkpoint"]).resolve()
        if selection.get("selected_checkpoint_sha256") != sha256_file(selected):
            raise ValueError("HHC selection manifest checkpoint hash is stale")
        return selected, artifact(selection_path)
    if not args.hhc_mgt_checkpoint:
        raise ValueError(
            "provide --hhc-selection-manifest (formal run) or "
            "--hhc-mgt-checkpoint (development audit)"
        )
    return Path(args.hhc_mgt_checkpoint).resolve(), None


def orchestrate(args):
    hhc_checkpoint, selection_record = resolve_hhc_checkpoint(args)
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="model-capacity-") as temporary:
        component_outputs = {}
        for component in ("hhc", "intermask", "coshmdm"):
            output = Path(temporary) / f"{component}.json"
            command = [
                sys.executable, str(script),
                "--component", component,
                "--component-output", str(output),
                "--hhc-root", str(Path(args.hhc_root).resolve()),
                "--hhc-vq-checkpoint", str(Path(args.hhc_vq_checkpoint).resolve()),
                "--hhc-vq-contract", args.hhc_vq_contract,
                "--hhc-mgt-checkpoint", str(hhc_checkpoint),
                "--intermask-root", str(Path(args.intermask_root).resolve()),
                "--coshmdm-root", str(Path(args.coshmdm_root).resolve()),
                "--coshmdm-checkpoint", str(Path(args.coshmdm_checkpoint).resolve()),
            ]
            subprocess.run(command, check=True)
            with output.open(encoding="utf-8") as handle:
                component_outputs[component] = json.load(handle)

    payload = {
        "schema": "hhc_model_capacity_audit_v2",
        "status": "pass",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audit_script": artifact(script),
        "counting_rule": (
            "unique torch.nn.Parameter object numel over checkpoint-validated "
            "instantiated modules; shared HHC VQ counted once; frozen text "
            "encoder parameters reported separately"
        ),
        "hhc": component_outputs["hhc"],
        "intermask": component_outputs["intermask"],
        "coshmdm": component_outputs["coshmdm"],
    }
    if selection_record is not None:
        payload["hhc"]["selection_manifest"] = selection_record
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/model_capacity_audit.json")
    parser.add_argument("--hhc-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--hhc-vq-checkpoint",
        default="checkpoints/vq_precleanup_raw12288/net_175000.pth",
    )
    parser.add_argument(
        "--hhc-vq-contract",
        choices=("current", "original_8192", "precleanup_raw12288"),
        default="precleanup_raw12288",
    )
    parser.add_argument("--hhc-selection-manifest")
    parser.add_argument("--hhc-mgt-checkpoint")
    parser.add_argument(
        "--intermask-root", default="/mnt/8T-HDD/wenning/InterMask"
    )
    parser.add_argument(
        "--coshmdm-root", default="/mnt/8T-HDD/wenning/coshmdm-code"
    )
    parser.add_argument(
        "--coshmdm-checkpoint",
        default=(
            "/mnt/8T-HDD/wenning/coshmdm-code/checkpoints/"
            "CoShMDM-InterX-fair-statsfix-seed123-bs32x2/model/last.ckpt"
        ),
    )
    parser.add_argument("--component", choices=("hhc", "intermask", "coshmdm"))
    parser.add_argument("--component-output")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.component:
        if not parsed.component_output:
            raise ValueError("--component-output is required in component mode")
        component_main(parsed)
    else:
        orchestrate(parsed)
