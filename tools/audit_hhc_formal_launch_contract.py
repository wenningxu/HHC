#!/usr/bin/env python
"""Freeze and audit the post-repair formal HHC launcher before execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/run_formal_hhc_after_mgt_repair.sh"
STATE_RESOLVER = ROOT / "tools/resolve_mgt_training_state.py"
SOURCE_FREEZER = ROOT / "tools/freeze_mgt_training_sources.py"
TRAIN_ENTRYPOINT = ROOT / "train/train_t2i_trans.py"
VALIDATION_IMPLEMENTATION = ROOT / "utils/eval_trans.py"
FAIR_EXPORTER = ROOT / "tools/export_hhc_official_interx.py"
OUTPUT = ROOT / "output/hhc_formal_launch_contract_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-snapshot-manifest', type=Path)
    args = parser.parse_args()

    training_entrypoint = TRAIN_ENTRYPOINT
    validation_implementation = VALIDATION_IMPLEMENTATION
    source_snapshot_manifest = None
    if args.source_snapshot_manifest is not None:
        snapshot_path = args.source_snapshot_manifest.resolve()
        source_snapshot_manifest = json.loads(
            snapshot_path.read_text(encoding='utf-8')
        )
        if (
            source_snapshot_manifest.get('schema')
            != 'hhc_mgt_launch_source_snapshot_v1'
            or source_snapshot_manifest.get('status') != 'pass'
            or source_snapshot_manifest.get('source_count') != 14
        ):
            raise ValueError('invalid formal HHC source snapshot manifest')
        snapshot_sources = source_snapshot_manifest.get('sources', {})
        selected_paths = {}
        for relative in ('train/train_t2i_trans.py', 'utils/eval_trans.py'):
            record = snapshot_sources.get(relative, {})
            path = Path(record.get('path', ''))
            if (
                not path.is_file()
                or path.stat().st_size != record.get('bytes')
                or sha256_file(path) != record.get('sha256')
            ):
                raise ValueError(f'invalid formal source snapshot: {relative}')
            selected_paths[relative] = path
        training_entrypoint = selected_paths['train/train_t2i_trans.py']
        validation_implementation = selected_paths['utils/eval_trans.py']

    source = LAUNCHER.read_text(encoding="utf-8")
    train_source = training_entrypoint.read_text(encoding="utf-8")
    validation_source = validation_implementation.read_text(encoding="utf-8")
    exporter_source = FAIR_EXPORTER.read_text(encoding="utf-8")
    required = [
        "tools/require_mgt_repair_gate.py --gate output/mgt_repair_gate.json",
        "tools/audit_historical_mgt_tokenizer_runtime.py",
        "--output output/historical_mgt_tokenizer_runtime_audit.json",
        "tools/audit_hhc_model_structure_runtime.py",
        "--output output/hhc_model_structure_runtime_audit.json",
        "tools/audit_hhc_training_mask_runtime.py",
        "--output output/hhc_training_mask_runtime_audit.json",
        "tools/audit_hhc_sampling_runtime.py",
        "--output output/hhc_sampling_runtime_audit.json",
        'vq_checkpoint="checkpoints/vq_precleanup_raw12288/net_175000.pth"',
        "tools/resolve_mgt_training_state.py",
        "tools/freeze_mgt_training_sources.py",
        'launch_source_snapshot_manifest="${run_dir}/launch_source_snapshot_manifest.json"',
        '--source-snapshot-manifest "${launch_source_snapshot_manifest}"',
        "--resume-training-ckpt",
        "--resume-in-place",
        'historical_codebook_source="/home/wenning/projects/MMM/checkpoints/vq/codebook"',
        'historical_contact_source="/home/wenning/projects/MMM/checkpoints/vq/contact_token_label"',
        "--token-cache-contract historical_mmm_exact",
        '--historical-codebook-source "${historical_codebook_source}"',
        '--historical-contact-source "${historical_contact_source}"',
        "--nb-code 12288",
        "--embed-dim-gpt 512 --n-head-gpt 8 --num-layers 9 --num-local-layer 9",
        "--batch-size 96 --total-iter 300000 --warm-up-iter 1000",
        "--lr 0.0002 --weight-decay 0.000001",
        "--contact --no-contact-conditioning --shape --paper-architecture",
        "--training-mask-contract paper_kinematic",
        "--training-loss-contract paper_masked_only",
        "--optimizer-contract paper_warmup_cosine",
        "--conditioning-contract paper_text_cfg",
        "--training-token-length-contract paper_50",
        "--training-validation-contract paper_isolated_common",
        "--sampler sample_intra_cas --temperature 0.5 --sampling-steps 50",
        "--cfg-scale 3.5 --contact-guidance-scale 1.0",
        'experiment="hhc_formal_raw12288_repaired_mgt_300k_seed123_v1"',
        "tools/audit_lightning_checkpoint_pair.py",
        "prevalidation-000020000.ckpt",
        "postvalidation-000020000.ckpt",
        "--expected-step 20000",
        "--require-fixed-pair-marker",
        "output/hhc_formal_raw12288_repaired_mgt_20k_pre_post_validation_pair_audit.json",
        "--expected-final-step 300000",
        "--hhc-vq-contract precleanup_raw12288 --hhc-mgt-contract current",
    ]
    forbidden = [
        '--token-cache-contract deterministic_raw_v1',
        "vq_paper",
        "net_150000.pth",
        "--nb-code 8096",
        "paper_mgt_precleanup_raw12288_300k_seed123\"",
        "body-controlled",
        "body_controlled",
    ]
    missing = [value for value in required if value not in source]
    present_forbidden = [value for value in forbidden if value in source]
    ordering = {
        "gate_before_cache": source.index("require_mgt_repair_gate.py")
        < source.index("train/train_t2i_trans.py --cache-only"),
        "historical_tokenizer_audit_before_cache": source.index(
            "audit_historical_mgt_tokenizer_runtime.py"
        ) < source.index("train/train_t2i_trans.py --cache-only"),
        "state_resolution_before_cache": source.index(
            "resolve_mgt_training_state.py"
        ) < source.index("train/train_t2i_trans.py --cache-only"),
        "sampling_runtime_audit_before_cache": source.index(
            "audit_hhc_sampling_runtime.py"
        ) < source.index("train/train_t2i_trans.py --cache-only"),
        "sampling_runtime_audit_before_training": source.index(
            "audit_hhc_sampling_runtime.py"
        ) < source.rindex("train/train_t2i_trans.py"),
        "training_mask_runtime_audit_before_cache": source.index(
            "audit_hhc_training_mask_runtime.py"
        ) < source.index("train/train_t2i_trans.py --cache-only"),
        "training_mask_runtime_audit_before_training": source.index(
            "audit_hhc_training_mask_runtime.py"
        ) < source.rindex("train/train_t2i_trans.py"),
        "model_structure_runtime_audit_before_cache": source.index(
            "audit_hhc_model_structure_runtime.py"
        ) < source.index("train/train_t2i_trans.py --cache-only"),
        "model_structure_runtime_audit_before_training": source.index(
            "audit_hhc_model_structure_runtime.py"
        ) < source.rindex("train/train_t2i_trans.py"),
        "cache_before_training": source.index("train/train_t2i_trans.py --cache-only")
        < source.rindex("train/train_t2i_trans.py"),
        "training_before_selection": source.rindex("train/train_t2i_trans.py")
        < source.index("select_best_mgt_checkpoint.py"),
        "training_before_validation_pair": source.rindex("train/train_t2i_trans.py")
        < source.index("audit_lightning_checkpoint_pair.py"),
        "selection_before_validation_pair": source.index(
            "select_best_mgt_checkpoint.py"
        ) < source.index("audit_lightning_checkpoint_pair.py"),
        "training_before_source_snapshot": source.rindex(
            "train/train_t2i_trans.py"
        ) < source.index("freeze_mgt_training_sources.py"),
        "selection_before_source_snapshot": source.index(
            "select_best_mgt_checkpoint.py"
        ) < source.index("freeze_mgt_training_sources.py"),
        "source_snapshot_before_paper_audit": source.index(
            "freeze_mgt_training_sources.py"
        ) < source.index("audit_hhc_paper_training_contract.py"),
        "source_snapshot_before_snapshot_bound_launch_audit": source.index(
            "freeze_mgt_training_sources.py"
        ) < source.rindex("audit_hhc_formal_launch_contract.py"),
        "validation_pair_before_fair_evaluation": source.index(
            "audit_lightning_checkpoint_pair.py"
        ) < source.index("run_fair_interx_method.py"),
        "selection_before_fair_evaluation": source.index("select_best_mgt_checkpoint.py")
        < source.index("run_fair_interx_method.py"),
    }
    validation_export_sampling_parity = {
        "validation_selects_configured_sampler": all(
            marker in train_source for marker in (
                "validation_sampler = (",
                "self.args.sampler if isolated_validation",
                "sampler=validation_sampler",
            )
        ),
        "validation_passes_all_sampling_parameters": all(
            marker in train_source for marker in (
                "CFG=self.args.cfg_scale",
                "temperature=self.args.temperature",
                "sampling_steps=self.args.sampling_steps",
                "contact_guidance_scale=self.args.contact_guidance_scale",
                "contact_map_path=self.args.contact_map_path",
            )
        ),
        "validation_uses_contact_schedule_only_at_sampling": all(
            marker in train_source for marker in (
                "self.contact = args.contact_conditioning",
                "contact=self.args.contact",
            )
        ),
        "validation_rng_is_fixed_and_observational": all(
            marker in train_source for marker in (
                "preserving_rng_state() if isolated_validation",
                "seed_evaluation(self.args.seed + 500_000)",
            )
        ),
        "validation_uses_raw_interx_normalization": (
            "MotionNormalizerTorchInterX('raw')" in validation_source
        ),
        "validation_downsamples_contact_by_four": (
            "downsample_contact_mask(contact_mask, factor=4)" in validation_source
        ),
        "validation_forwards_cascade_contact_bias": all(
            marker in validation_source for marker in (
                "type=sampler",
                "max_steps=sampling_steps",
                "vocab_contact_map=vocab_contact_map",
                "contact_guidance_scale=contact_guidance_scale",
                "temperature=temperature",
            )
        ),
        "fair_export_uses_raw_interx_normalization": all(
            marker in exporter_source for marker in (
                "MotionNormalizerTorchInterX(",
                "'raw', stats_dir=args.stats_dir, stats_prefix=args.stats_prefix",
            )
        ),
        "fair_export_forwards_same_sampling_parameters": all(
            marker in exporter_source for marker in (
                "type='sample_intra_cas'",
                "max_steps=args.sampling_steps",
                "CFG=args.cfg_scale",
                "contact_guidance_scale=args.contact_guidance_scale",
                "temperature=args.temperature",
            )
        ),
        "formal_launch_uses_one_shared_train_only_contact_map": all(
            marker in source for marker in (
                'cache_root="checkpoints/vq_precleanup_raw12288_formal_mgt_v1"',
                '--hhc-contact-map "${cache_root}/code_contact_probs.npy"',
                '--hhc-contact-map-contract paper_formula_train_only',
            )
        ),
        "formal_launch_exports_contact_and_no_contact_rows": all(
            marker in source for marker in (
                "for guidance in no_contact contact",
                "--hhc-contact-timing-source request_oracle --contact-guidance",
                "--hhc-contact-timing-source none --no-contact-guidance",
            )
        ),
    }
    status = (
        "pass"
        if not missing
        and not present_forbidden
        and all(ordering.values())
        and all(validation_export_sampling_parity.values())
        else "fail"
    )
    payload = {
        "schema": "hhc_formal_launch_contract_v1",
        "status": status,
        "launcher": {
            "path": str(LAUNCHER),
            "sha256": sha256_file(LAUNCHER),
            "bytes": LAUNCHER.stat().st_size,
        },
        "state_resolver": {
            "path": str(STATE_RESOLVER),
            "sha256": sha256_file(STATE_RESOLVER),
            "bytes": STATE_RESOLVER.stat().st_size,
        },
        "source_freezer": {
            "path": str(SOURCE_FREEZER),
            "sha256": sha256_file(SOURCE_FREEZER),
            "bytes": SOURCE_FREEZER.stat().st_size,
        },
        "sampling_parity_sources": {
            "training_entrypoint": {
                "path": str(training_entrypoint),
                "sha256": sha256_file(training_entrypoint),
                "bytes": training_entrypoint.stat().st_size,
            },
            "validation_implementation": {
                "path": str(validation_implementation),
                "sha256": sha256_file(validation_implementation),
                "bytes": validation_implementation.stat().st_size,
            },
            "fair_exporter": {
                "path": str(FAIR_EXPORTER),
                "sha256": sha256_file(FAIR_EXPORTER),
                "bytes": FAIR_EXPORTER.stat().st_size,
            },
        },
        "sampling_parity_source_scope": (
            "immutable_launch_snapshot"
            if source_snapshot_manifest is not None else "live_source_tree"
        ),
        "source_snapshot_manifest": (
            {
                "path": str(args.source_snapshot_manifest.resolve()),
                "sha256": sha256_file(args.source_snapshot_manifest.resolve()),
                "bytes": args.source_snapshot_manifest.resolve().stat().st_size,
            }
            if source_snapshot_manifest is not None else None
        ),
        "required_contract_entries": required,
        "missing_contract_entries": missing,
        "forbidden_entries_present": present_forbidden,
        "ordering": ordering,
        "validation_export_sampling_parity": validation_export_sampling_parity,
        "training_entrypoint_count": source.count("train/train_t2i_trans.py"),
        "evaluation_labels": [
            "hhc_formal_raw12288_repaired_mgt_common50_no_contact",
            "hhc_formal_raw12288_repaired_mgt_common50_contact",
        ],
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
