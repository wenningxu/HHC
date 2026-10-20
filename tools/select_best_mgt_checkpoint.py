#!/usr/bin/env python
"""Select the lowest fixed-validation-FID MGT checkpoint after a complete run."""

import argparse
import glob
import hashlib
import json
import math
import os
import re

import torch

try:
    from tools.audit_lightning_checkpoint import tensor_audit
except ModuleNotFoundError:  # Direct `python tools/select_best_...py` execution.
    from audit_lightning_checkpoint import tensor_audit


MODEL_PATTERN = re.compile(r"model-(\d+)-([-+0-9.eE]+)\.ckpt$")
LATEST_PATTERN = re.compile(r"latest-(\d+)\.ckpt$")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_completed_run(run_directories, expected_final_step):
    latest_by_run = {}
    completed_runs = []
    for directory in run_directories:
        latest = []
        for path in glob.glob(os.path.join(directory, "latest-*.ckpt")):
            match = LATEST_PATTERN.search(os.path.basename(path))
            if match:
                latest.append((int(match.group(1)), os.path.abspath(path)))
        latest_by_run[directory] = latest
        final_paths = [
            path for step, path in latest if step == expected_final_step
        ]
        if len(final_paths) > 1:
            raise RuntimeError(
                f"multiple final checkpoints in completed run {directory}: "
                f"{final_paths}"
            )
        if len(final_paths) == 1:
            completed_runs.append((directory, final_paths[0]))
    if len(completed_runs) != 1:
        raise RuntimeError(
            f"expected exactly one run demonstrably complete at step "
            f"{expected_final_step}, got {completed_runs}; "
            f"latest checkpoints by run={latest_by_run}"
        )
    run_directory, final_checkpoint = completed_runs[0]
    return run_directory, final_checkpoint, latest_by_run


def audit_final_checkpoint(path, expected_step):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    global_step = int(checkpoint.get('global_step', -1))
    if global_step != expected_step:
        raise ValueError(
            f'final checkpoint filename says {expected_step} but state has '
            f'global_step={global_step}'
        )
    state_dict = checkpoint.get('state_dict')
    optimizer_states = checkpoint.get('optimizer_states')
    schedulers = checkpoint.get('lr_schedulers')
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError('final checkpoint has no model state')
    if not isinstance(optimizer_states, list) or not optimizer_states:
        raise ValueError('final checkpoint has no optimizer state')
    if not isinstance(schedulers, list) or not schedulers:
        raise ValueError('final checkpoint has no scheduler state')
    model_state = tensor_audit(state_dict)
    optimizer_state = tensor_audit(optimizer_states)
    if not model_state['all_floating_tensors_finite']:
        raise ValueError('final checkpoint model contains NaN or Inf')
    if not optimizer_state['all_floating_tensors_finite']:
        raise ValueError('final checkpoint optimizer contains NaN or Inf')
    return {
        'status': 'pass',
        'global_step': global_step,
        'epoch': int(checkpoint.get('epoch', -1)),
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'optimizer_count': len(optimizer_states),
        'optimizer_parameter_state_entries': int(sum(
            len(optimizer.get('state', {})) for optimizer in optimizer_states
        )),
        'lr_scheduler_count': len(schedulers),
        'lr_scheduler_progress': [
            {
                key: scheduler[key]
                for key in ('last_epoch', '_step_count') if key in scheduler
            }
            for scheduler in schedulers
        ],
        'has_loop_state': bool(checkpoint.get('loops')),
    }


def exact_validation_fid(path, expected_step, filename_fid):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    checkpoint_step = int(checkpoint.get('global_step', -1))
    if checkpoint_step != expected_step:
        raise ValueError(
            f'validation checkpoint step mismatch for {path}: '
            f'{checkpoint_step} != {expected_step}'
        )
    callback_records = [
        state for state in checkpoint.get('callbacks', {}).values()
        if isinstance(state, dict) and state.get('monitor') == 'val/FID'
    ]
    if len(callback_records) != 1:
        raise ValueError(
            f'{path}: expected one val/FID ModelCheckpoint state, got '
            f'{len(callback_records)}'
        )
    score = callback_records[0].get('current_score')
    if score is None:
        raise ValueError(f'{path}: val/FID callback has no current_score')
    score = float(score.item() if torch.is_tensor(score) else score)
    if not math.isfinite(score) or score < 0:
        raise ValueError(f'{path}: invalid exact validation FID {score}')
    if abs(score - filename_fid) > 5.1e-5:
        raise ValueError(
            f'{path}: exact validation FID {score} does not round to '
            f'filename value {filename_fid}'
        )
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_directories = sorted(
        os.path.abspath(path)
        for path in glob.glob(args.run_glob) if os.path.isdir(path)
    )
    if not run_directories:
        raise RuntimeError(
            f"no MGT run directories matched {args.run_glob!r}"
        )
    run_directory, final_checkpoint, latest_by_run = discover_completed_run(
        run_directories, args.expected_final_step
    )

    candidates = []
    for path in glob.glob(os.path.join(run_directory, "model-*.ckpt")):
        match = MODEL_PATTERN.search(os.path.basename(path))
        if match is None:
            continue
        step = int(match.group(1))
        filename_validation_fid = float(match.group(2))
        if not (0 <= step <= args.expected_final_step):
            raise ValueError(f"checkpoint step outside the completed run: {path}")
        if not (filename_validation_fid >= 0
                and filename_validation_fid < float("inf")):
            raise ValueError(f"invalid validation FID in checkpoint name: {path}")
        absolute_path = os.path.abspath(path)
        validation_fid = exact_validation_fid(
            absolute_path, step, filename_validation_fid
        )
        candidates.append((
            validation_fid, step, absolute_path, filename_validation_fid
        ))
    if not candidates:
        raise RuntimeError(f"no validation-FID checkpoints found in {run_directory}")

    validation_fid, step, selected, filename_validation_fid = min(candidates)
    training_manifest = os.path.join(run_directory, "training_manifest.json")
    if not os.path.isfile(training_manifest):
        raise FileNotFoundError(training_manifest)
    with open(training_manifest, encoding='utf-8') as handle:
        training = json.load(handle)
    resolved = training.get('resolved_arguments', {})
    expected_validation_arguments = {
        'sampler': 'sample_intra_cas',
        'sampling_steps': 50,
        'temperature': 0.5,
        'cfg_scale': 3.5,
        'contact_guidance_scale': 1.0,
        'contact': True,
        'contact_conditioning': False,
        'eval_iter': 10000,
    }
    mismatched_validation_arguments = {
        key: {'expected': expected, 'actual': resolved.get(key)}
        for key, expected in expected_validation_arguments.items()
        if resolved.get(key) != expected
    }
    if mismatched_validation_arguments:
        raise ValueError(
            'MGT checkpoint selection validation contract changed: '
            f'{mismatched_validation_arguments}'
        )
    final_checkpoint_sha256 = sha256_file(final_checkpoint)
    final_state_audit = audit_final_checkpoint(
        final_checkpoint, args.expected_final_step
    )
    result = {
        "run_directory": run_directory,
        "matched_run_directories": run_directories,
        "ignored_incomplete_run_directories": [
            directory for directory in run_directories
            if directory != run_directory
        ],
        "run_complete": True,
        "expected_final_step": args.expected_final_step,
        "final_latest_checkpoint": final_checkpoint,
        "final_latest_checkpoint_bytes": os.path.getsize(final_checkpoint),
        "final_latest_checkpoint_sha256": final_checkpoint_sha256,
        "final_state_audit": final_state_audit,
        "selection_metric": "minimum fixed-protocol validation FID",
        "test_metrics_used_for_selection": False,
        "validation_inference_contract": {
            "contact_timing_source": "validation_dataset_annotation_oracle",
            "contact_timing_is_test_data": False,
            "contact_as_transformer_training_input": False,
            **expected_validation_arguments,
        },
        "retained_validation_candidates": [
            {
                "path": path, "step": candidate_step,
                "validation_fid": candidate_fid,
                "filename_validation_fid": candidate_filename_fid,
            }
            for candidate_fid, candidate_step, path, candidate_filename_fid
            in sorted(candidates)
        ],
        "selected_checkpoint": selected,
        "selected_checkpoint_bytes": os.path.getsize(selected),
        "selected_checkpoint_sha256": sha256_file(selected),
        "selected_step": step,
        "selected_validation_fid": validation_fid,
        "selected_filename_validation_fid": filename_validation_fid,
        "training_manifest": os.path.abspath(training_manifest),
        "training_manifest_sha256": sha256_file(training_manifest),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = os.path.abspath(args.output) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
