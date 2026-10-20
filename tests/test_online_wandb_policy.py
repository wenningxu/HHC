import ast
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.apply_post_formal_online_wandb_metrics import (
    online_wandb_metrics_already_applied,
    transform_sources,
)
from tools.audit_wandb_online_metric_history import (
    PROFILE_RUN_BINDINGS,
    history_coverage_checks,
    metric_history,
    resolve_optimizer_step_axis,
)
from tools.inspect_local_wandb_progress import summarize_training_metric_progress
from utils.training_telemetry import (
    GradientClipTelemetry,
    parameter_and_gradient_telemetry,
)


def test_post_formal_online_wandb_transform_is_parseable_and_complete():
    current = (
        (ROOT / "options/option_transformer.py").read_text(encoding="utf-8"),
        (ROOT / "train/train_t2i_trans.py").read_text(encoding="utf-8"),
        (ROOT / "utils/eval_trans.py").read_text(encoding="utf-8"),
    )
    already_applied = online_wandb_metrics_already_applied(current[1])
    options, train, evaluation = (
        current if already_applied else transform_sources(*current)
    )
    for label, source in (
        ("options", options), ("train", train), ("evaluation", evaluation)
    ):
        ast.parse(source, filename=label)
    policy = json.loads((
        ROOT / "configs/online_wandb_training_policy.json"
    ).read_text(encoding="utf-8"))
    combined = train + (ROOT / "utils/training_telemetry.py").read_text()
    for metric in (
        policy["required_training_metrics"]
        + policy["required_validation_metrics"]
    ):
        assert metric in combined
    assert policy["logging_interval_optimizer_steps"] == 50
    assert policy["parameter_scope"] == "all trainable parameters"
    assert "before gradient clipping" in policy["gradient_measurement_point"]
    assert "before optimizer step" in policy["weight_measurement_point"]
    assert "train/modules/<name>/" in policy["major_module_norm_policy"]
    assert policy["major_module_norm_effective_from"] == "2026-09-02"
    assert "post-clip global L2 norm" in policy[
        "gradient_clipping_observability"
    ]
    assert set(policy["conditionally_required_training_metrics"][
        "when_gradient_clipping_is_enabled"
    ]) == {
        "train/gradient_l2_norm_post_clip",
        "train/gradient_clip_threshold",
        "train/gradient_clip_triggered",
        "train/gradient_clip_hit_rate",
    }
    assert "W&B _step is forbidden" in policy["training_progress_axis"]
    assert "final required training-metric event" in (
        policy["validation_coverage_policy"]
    )
    norm_metrics = {
        "train/gradient_l2_norm",
        "train/gradient_max_abs",
        "train/weight_l2_norm",
        "train/weight_max_abs",
        "train/gradient_to_weight_l2_ratio",
        "train/gradient_value_count",
        "train/weight_value_count",
        "train/nonfinite_gradients",
        "train/nonfinite_weights",
    }
    legacy_profile_name = "diffusion_regression_20260831_launch"
    for profile_name, profile in policy["metric_profiles"].items():
        if profile_name == legacy_profile_name:
            assert {
                "train/gradient_l2_norm",
                "train/gradient_max_abs",
                "train/weight_l2_norm",
                "train/weight_max_abs",
                "train/nonfinite_gradients",
                "train/nonfinite_weights",
            } <= set(profile["required_training_metrics"])
            assert "2026-08-31" in profile["scope"]
        else:
            assert norm_metrics <= set(profile["required_training_metrics"])
    assert PROFILE_RUN_BINDINGS == {
        legacy_profile_name: {"project": "HHC", "run_id": "kdz5s3ra"},
    }
    assert "default='online'" in options
    assert "args.wandb_run_id = wandb.util.generate_id()" in train
    assert "'wandb_project': 'NewInter'" in train
    assert "'metric_profile': 'masked_token_classification'" in train
    assert "all newly launched MGT training requires --wandb-mode online" in train
    assert train.index(
        "acc_all = right_token.sum()"
    ) < train.index(
        "if next_step % self.telemetry_interval_optimizer_steps == 0:"
    )
    assert "cls_pred.detach().argmax(dim=-1)" in train
    assert "return right_num * 100.0 / total_num" in train
    assert "args.cache_only or args.generate_only or args.evaluate_only" in train
    assert "resume='allow' if args.wandb_mode == 'online' else None" in train
    assert "return_current_metrics=True" in train
    assert "return_current_metrics=False" in evaluation
    assert "'train/optimizer_step': float(next_step)" in train
    assert online_wandb_metrics_already_applied(train) is True
    assert online_wandb_metrics_already_applied(current[1]) is already_applied


def test_get_acc_is_a_type_stable_scalar_tensor_for_empty_masks():
    source = (ROOT / "train/train_t2i_trans.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_acc"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "get_acc", "exec"),
        namespace,
    )
    logits = torch.zeros((1, 2, 3))
    target = torch.zeros((1, 2), dtype=torch.long)
    empty_mask = torch.zeros((1, 2), dtype=torch.bool)
    result = namespace["get_acc"](logits, target, empty_mask)
    assert torch.is_tensor(result)
    assert result.ndim == 0
    assert result.item() == 0.0


def test_vq_training_entrypoint_enforces_online_full_process_telemetry():
    options = (ROOT / "options/option_vq.py").read_text(encoding="utf-8")
    train = (ROOT / "train/train_vq_interx.py").read_text(encoding="utf-8")
    ast.parse(options, filename="options/option_vq.py")
    ast.parse(train, filename="train/train_vq_interx.py")
    policy = json.loads((
        ROOT / "configs/online_wandb_training_policy.json"
    ).read_text(encoding="utf-8"))
    profile = policy["metric_profiles"]["vq_autoencoder"]
    canonical = policy["metric_profiles"]["vq_autoencoder_body_root_hands"]
    assert "'metric_profile': 'vq_autoencoder_body_root_hands'" in train
    for part, index, aggregate in [('body', 0, 'u'), ('root', 1, 'r'), ('hands', 2, 'l')]:
        metric = f'train/perplexity_{part}'
        assert metric in canonical['required_training_metrics']
        assert f"'{metric}': float(perplexity[{index}].detach())" in train
        assert f'"{metric}": avg_metrics["perplexity_{aggregate}"]' in train
    assert set(canonical['required_training_metrics']) == (
        set(profile['required_training_metrics'])
        - {'train/perplexity_right_hand', 'train/perplexity_left_hand'}
        | {'train/perplexity_root', 'train/perplexity_hands'}
    )
    for metric in (
        profile["required_training_metrics"]
        + profile["required_validation_metrics"]
    ):
        assert metric in train or metric in (
            "train/gradient_l2_norm",
            "train/gradient_max_abs",
            "train/weight_l2_norm",
            "train/weight_max_abs",
            "train/gradient_to_weight_l2_ratio",
            "train/gradient_value_count",
            "train/weight_value_count",
            "train/nonfinite_gradients",
            "train/nonfinite_weights",
        )
    assert "default='online'" in options
    assert "all newly launched VQ training requires --wandb-mode online" in train
    assert "id=args.wandb_run_id" in train
    assert "resume='allow'" in train
    assert "parameter_and_gradient_telemetry(" in train
    assert "telemetry_interval_optimizer_steps = 50" in train
    assert "'train/optimizer_step': float(optimizer_step)" in train
    assert "fail_on_nonfinite=False" in train
    assert "'final_optimizer_step': main_update_offset + args.total_iter" in train
    contract_audit = (
        ROOT / "tools/audit_online_wandb_training_contract.py"
    ).read_text(encoding="utf-8")
    assert '"vq_autoencoder"' in contract_audit
    assert '"wandb_logging_backend_present"' in contract_audit
    assert 'wandb\\.init' in contract_audit
    assert 'wandb\\.log' in contract_audit


def test_shell_training_launchers_never_request_offline_wandb():
    """Keep stale queue scripts from bypassing the online-only entrypoints."""
    offenders = []
    for launcher in sorted((ROOT / "tools").glob("*.sh")):
        for line_number, line in enumerate(
            launcher.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "--wandb-mode offline" in line:
                offenders.append(f"{launcher.relative_to(ROOT)}:{line_number}")
    assert not offenders, "offline training launchers: " + ", ".join(offenders)


def test_parameter_and_gradient_telemetry_reports_exact_finite_norms():
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[3.0, 4.0]]))
    module.weight.grad = torch.tensor([[5.0, 12.0]])
    metrics = parameter_and_gradient_telemetry(module)
    assert float(metrics["train/weight_l2_norm"]) == 5.0
    assert float(metrics["train/gradient_l2_norm"]) == 13.0
    assert float(metrics["train/weight_max_abs"]) == 4.0
    assert float(metrics["train/gradient_max_abs"]) == 12.0
    assert metrics["train/nonfinite_weights"] == 0.0
    assert metrics["train/nonfinite_gradients"] == 0.0
    assert abs(float(metrics["train/gradient_to_weight_l2_ratio"]) - 2.6) < 1e-6
    assert metrics["train/gradient_value_count"] == 2.0
    assert metrics["train/weight_value_count"] == 2.0


def test_parameter_and_gradient_telemetry_reports_top_level_modules():
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 1, bias=False),
        torch.nn.Linear(1, 1, bias=False),
    )
    with torch.no_grad():
        module[0].weight.copy_(torch.tensor([[3.0, 4.0]]))
        module[1].weight.copy_(torch.tensor([[2.0]]))
    module[0].weight.grad = torch.tensor([[5.0, 12.0]])
    module[1].weight.grad = torch.tensor([[6.0]])
    metrics = parameter_and_gradient_telemetry(module)
    assert float(metrics["train/modules/0/weight_l2_norm"]) == 5.0
    assert float(metrics["train/modules/0/gradient_l2_norm"]) == 13.0
    assert float(metrics["train/modules/1/weight_l2_norm"]) == 2.0
    assert float(metrics["train/modules/1/gradient_l2_norm"]) == 6.0
    assert metrics["train/modules/0/nonfinite_gradients"] == 0.0


def test_parameter_and_gradient_telemetry_fails_on_nonfinite_values():
    module = torch.nn.Linear(1, 1, bias=False)
    module.weight.grad = torch.tensor([[float("nan")]])
    try:
        parameter_and_gradient_telemetry(module)
    except FloatingPointError as error:
        assert "nonfinite training telemetry" in str(error)
    else:
        raise AssertionError("non-finite gradient was accepted")


def test_parameter_and_gradient_telemetry_can_report_before_fail_closed():
    module = torch.nn.Linear(1, 1, bias=False)
    module.weight.grad = torch.tensor([[float("nan")]])
    metrics = parameter_and_gradient_telemetry(
        module, fail_on_nonfinite=False
    )
    assert metrics["train/nonfinite_gradients"] == 1.0
    assert metrics["train/nonfinite_weights"] == 0.0


def test_gradient_clip_telemetry_reports_postclip_norm_and_cumulative_hits():
    module = torch.nn.Linear(2, 1, bias=False)
    clip = GradientClipTelemetry(1.0)
    module.weight.grad = torch.tensor([[3.0, 4.0]])
    metrics = clip.clip(module, emit_metrics=True)
    assert abs(float(metrics["train/gradient_l2_norm_post_clip"]) - 1.0) < 1e-6
    assert metrics["train/gradient_clip_threshold"] == 1.0
    assert metrics["train/gradient_clip_triggered"] == 1.0
    assert metrics["train/gradient_clip_hit_rate"] == 1.0

    module.weight.grad = torch.tensor([[0.3, 0.4]])
    metrics = clip.clip(module, emit_metrics=True)
    assert abs(float(metrics["train/gradient_l2_norm_post_clip"]) - 0.5) < 1e-6
    assert metrics["train/gradient_clip_triggered"] == 0.0
    assert metrics["train/gradient_clip_hit_rate"] == 0.5


def test_timotion_warmup_audit_does_not_expand_with_the_live_run():
    source = (ROOT / "tools/audit_wandb_warmup_stability.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source, filename="tools/audit_wandb_warmup_stability.py")
    assert "rows[:expected_events]" in source
    assert "expected_events = warmup_optimizer_steps // interval" in source
    assert '"metric_window": "first telemetry events through warm-up only"' in source
    assert '"full_run_metrics"' in source
    assert '"full_required_training_metrics"' in source
    assert '"training_metric_presence"' in source
    assert '"full_validation_metrics"' in source
    assert '"validation_metric_presence"' in source
    assert "finite_histories(" in source
    assert "SPARSE_TELEMETRY_METRICS" in source
    assert "for metric_group in (sparse_training, dense_training)" in source
    assert 'metric != "val/loss"' in source
    assert 'finite_event_histories(run, ("val/loss",))' in source
    assert 'profile["required_training_metrics"]' in source
    assert '"all_profile_training_histories_present_and_finite"' in source
    assert "W&B internal step is event" in source
    assert 'for row in warmup_histories[metric]' in source
    assert "resolve_optimizer_step_axis(run)" in source
    assert '"configured_optimizer_steps"' in source
    assert 'resolved_arguments.get("warm_up_iter", 0)' in source
    assert '"warmup_optimizer_steps"' in source
    assert 'completed_optimizer_step >= warmup_optimizer_steps' in source
    assert '"optimizer_step_axis": optimizer_step_axis' in source
    assert '"last_optimizer_step"' in source


def test_mgt_warmup_audit_uses_exact_optimizer_steps_and_reports_max_abs():
    source = (ROOT / "tools/audit_mgt_wandb_warmup.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source, filename="tools/audit_mgt_wandb_warmup.py")
    assert 'axis == "train/optimizer_step"' in source
    assert 'steps == expected_steps' in source
    assert 'warmup_steps % interval' in source
    assert '"nonfinite_counts_zero"' in source
    assert '"weight_l2_relative_span_bounded"' in source
    assert '"observed_weight_max_abs_relative_span"' in source
    assert '"reported_without_a_hard_drift_threshold' in source
    assert '"metric_window": "exact optimizer steps through warm-up only"' in source
    assert '"full_run_metrics"' in source
    assert '"full_validation_metrics"' in source
    assert '"validation_metric_presence"' in source
    assert "require_optimizer_axis=False" in source
    assert "W&B internal step is event" in source
    assert 'summarize_metric(rows)' in source
    assert "return None" in source
    launcher = (
        ROOT / "tools/run_hhc_original38_token_window_control.sh"
    ).read_text(encoding="utf-8")
    assert "tools/freeze_mgt_training_sources.py" in launcher
    assert launcher.index("tools/freeze_mgt_training_sources.py") < launcher.index(
        "tools/audit_mgt_wandb_warmup.py"
    )
    assert "tools/audit_mgt_wandb_warmup.py" in launcher
    assert launcher.index("tools/audit_mgt_wandb_warmup.py") < launcher.index(
        "tools/audit_wandb_online_metric_history.py"
    )


def test_completed_wandb_audit_requires_every_metric_through_final_step():
    profile = {
        "required_training_metrics": [
            "train/loss",
            "train/gradient_l2_norm",
            "train/nonfinite_gradients",
            "train/nonfinite_weights",
        ],
        "required_validation_metrics": ["val/loss", "val/FID"],
    }
    summaries = {
        metric: {
            "last_step": 100,
            "last_wandb_step": 100,
            "minimum": 0.0,
            "maximum": 0.0,
        }
        for metric in (
            profile["required_training_metrics"]
            + profile["required_validation_metrics"]
        )
    }
    checks = history_coverage_checks(summaries, profile, 100)
    assert all(checks.values())

    summaries["train/gradient_l2_norm"]["last_step"] = 50
    checks = history_coverage_checks(summaries, profile, 100)
    assert checks["all_training_metrics_reach_expected_step"] is False

    summaries["train/gradient_l2_norm"]["last_step"] = 100
    summaries["val/FID"]["last_wandb_step"] = 99
    checks = history_coverage_checks(summaries, profile, 100)
    assert checks[
        "all_validation_metrics_follow_final_training_event"
    ] is False


def test_completed_wandb_audit_rejects_any_nonfinite_count():
    profile = {
        "required_training_metrics": [
            "train/nonfinite_gradients", "train/nonfinite_weights",
        ],
        "required_validation_metrics": ["val/loss"],
    }
    summaries = {
        metric: {
            "last_step": 100,
            "last_wandb_step": 100,
            "minimum": 0.0,
            "maximum": 0.0,
        }
        for metric in (
            profile["required_training_metrics"]
            + profile["required_validation_metrics"]
        )
    }
    summaries["train/nonfinite_gradients"]["maximum"] = 1.0
    checks = history_coverage_checks(summaries, profile, 100)
    assert checks["nonfinite_counts_zero"] is False


def test_wandb_history_uses_optimizer_axis_instead_of_internal_step():
    class FakeRun:
        def scan_history(self, keys, page_size):
            assert "_step" in keys
            if "train/optimizer_step" in keys:
                return iter([])
            assert "trainer/global_step" in keys
            return iter([
                {
                    "_step": 6559,
                    "trainer/global_step": 5500,
                    "train/gradient_l2_norm": 5.7,
                },
            ])

    run = FakeRun()
    assert resolve_optimizer_step_axis(run) == "trainer/global_step"
    rows = metric_history(run, "train/gradient_l2_norm")
    assert rows == [{
        "optimizer_step": 5500,
        "wandb_step": 6559,
        "step_source": "trainer/global_step",
        "value": 5.7,
    }]


def test_single_snapshot_matches_per_metric_scans_and_rejects_nonfinite():
    rows = [
        {'_step': 1, 'train/optimizer_step': 20, 'train/loss': .3},
        {'_step': 2, 'train/optimizer_step': 40, 'train/loss': .2},
        {'_step': 3, 'val/fid': 1.5},
        {'_step': 4, 'train/optimizer_step': 60, 'train/loss': None},
    ]
    class FakeRun:
        def __init__(self):
            self.calls = 0
        def scan_history(self, keys=None, page_size=1000):
            self.calls += 1
            return iter(row for row in rows if keys is None or all(row.get(k) is not None for k in keys))
    run = FakeRun()
    expected_train = metric_history(run, 'train/loss')
    expected_val = metric_history(run, 'val/fid', require_optimizer_axis=False)
    run.calls = 0
    snapshot = list(run.scan_history(page_size=1000))
    assert resolve_optimizer_step_axis(run, snapshot) == 'train/optimizer_step'
    assert metric_history(run, 'train/loss', snapshot=snapshot) == expected_train
    assert metric_history(run, 'val/fid', require_optimizer_axis=False, snapshot=snapshot) == expected_val
    assert run.calls == 1
    snapshot.append({'_step': 5, 'train/optimizer_step': 80, 'train/loss': float('nan')})
    try:
        metric_history(run, 'train/loss', snapshot=snapshot)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('cached scan accepted nonfinite value')


def test_wandb_history_rejects_internal_step_as_only_axis():
    class FakeRun:
        def scan_history(self, keys, page_size):
            return iter([{"_step": 6559, "train/loss": 0.1}])

    try:
        metric_history(FakeRun(), "train/loss")
    except ValueError as error:
        assert "W&B _step is not an optimizer step" in str(error)
    else:
        raise AssertionError("W&B internal step was accepted as optimizer progress")


def test_validation_history_uses_internal_step_only_for_event_ordering():
    class FakeRun:
        def scan_history(self, keys, page_size):
            assert keys == ["_step", "val/FID"]
            return iter([{"_step": 7000, "val/FID": 0.3}])

    rows = metric_history(
        FakeRun(), "val/FID", require_optimizer_axis=False
    )
    assert rows[0]["optimizer_step"] is None
    assert rows[0]["wandb_step"] == 7000
    assert rows[0]["step_source"] == "wandb_event_order_only"


def test_live_progress_does_not_report_validation_microbatch_as_optimizer_step():
    metrics = {
        "train/loss": {
            "latest_progress": {"trainer/global_step": 57946, "epoch": 816},
        },
        "train/gradient_l2_norm": {
            "latest_progress": {"trainer/global_step": 57900, "epoch": 815},
        },
        "val/loss": {
            "latest_progress": {"trainer/global_step": 463734, "epoch": 814},
        },
    }
    summary = summarize_training_metric_progress(metrics)
    assert summary["optimizer_axis"] == "trainer/global_step"
    assert summary["minimum_covered_step"] == 57900
    assert summary["maximum_observed_step"] == 57946
    assert "val/loss" not in summary["per_metric_latest_step"]


if __name__ == "__main__":
    test_post_formal_online_wandb_transform_is_parseable_and_complete()
    test_get_acc_is_a_type_stable_scalar_tensor_for_empty_masks()
    test_vq_training_entrypoint_enforces_online_full_process_telemetry()
    test_shell_training_launchers_never_request_offline_wandb()
    test_parameter_and_gradient_telemetry_reports_exact_finite_norms()
    test_parameter_and_gradient_telemetry_reports_top_level_modules()
    test_parameter_and_gradient_telemetry_fails_on_nonfinite_values()
    test_parameter_and_gradient_telemetry_can_report_before_fail_closed()
    test_gradient_clip_telemetry_reports_postclip_norm_and_cumulative_hits()
    test_timotion_warmup_audit_does_not_expand_with_the_live_run()
    test_completed_wandb_audit_requires_every_metric_through_final_step()
    test_completed_wandb_audit_rejects_any_nonfinite_count()
    test_wandb_history_uses_optimizer_axis_instead_of_internal_step()
    test_wandb_history_rejects_internal_step_as_only_axis()
    test_validation_history_uses_internal_step_only_for_event_ordering()
    test_live_progress_does_not_report_validation_microbatch_as_optimizer_step()
