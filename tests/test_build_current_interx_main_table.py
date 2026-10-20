import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    "hhc_historical_plus_paper_mask_ratio_no_contact_common50",
    "hhc_historical_plus_paper_mask_ratio_contact_common50",
    "intermask_official_exact_common50",
    "interact2ar_official_fair_no_gt_first_frame",
    "coshmdm_interx_retrained_2daycap_seed123",
    "timotion_transformer_lpa_interx_retrained_fair",
)
METRICS = ("fid", "matching", "top1", "top2", "top3", "diversity", "multimodality")
PAIRED_METRICS = set(METRICS) - {"diversity"} | {"diversity_absolute_error"}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def artifact(path):
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def paired_payload(reference, reference_directory, comparators, fair_root):
    metric = {
        "wilcoxon_two_sided_p": 0.5,
        "holm_familywise_p": 1.0,
        "reference_advantage_bootstrap_95ci": [-0.1, 0.1],
    }
    return {
        "schema": "fair_interx_paired_method_comparison_v1",
        "reference": {
            "label": reference,
            "directory": str(fair_root / reference_directory),
        },
        "replications": 20,
        "replication_ids": list(range(20)),
        "pairing": "same replication ID and byte-verified frozen request NPZ",
        "test": "two-sided paired Wilcoxon signed-rank",
        "multiplicity_correction": "Holm-Bonferroni",
        "comparators": {
            alias: {
                "directory": str(fair_root / directory),
                "metrics": {name: metric for name in PAIRED_METRICS},
            }
            for alias, directory in comparators.items()
        },
    }


def test_current_table_requires_current_labels_and_builds_outputs(tmp_path):
    fair_root = tmp_path / "fair"
    values = [float(index) / 100 for index in range(20)]
    mean = sum(values) / len(values)
    for label in LABELS:
        summary_metrics = {
            name: {"values": values, "mean": mean, "ci95": 0.01}
            for name in METRICS
        }
        summary_metrics["diversity_gt"] = {
            "values": values, "mean": mean, "ci95": 0.01,
        }
        write_json(fair_root / label / "summary.json", {
            "replications": 20,
            "samples_per_replication": 1696,
            "metrics": summary_metrics,
        })
        write_json(fair_root / label / "run_manifest.json", {
            "label": label, "samples_per_replication": 1696,
            "replication_range": [0, 20],
        })
        for replication in range(20):
            write_json(
                fair_root / label / f"replication_{replication:02d}.json",
                {"replication": replication},
            )

    ordinary_audit = tmp_path / "ordinary.json"
    implementations = {
        "audit_and_rescore": ROOT / "tools/rescore_final_interx_ordinary.py",
        "semantic_scorer": ROOT / "tools/score_fair_interx_generations.py",
        "aggregator": ROOT / "tools/aggregate_fair_interx_metrics.py",
        "multimodality_aggregator": (
            ROOT / "tools/aggregate_fair_interx_multimodality_audits.py"
        ),
        "evaluator_options": ROOT / "checkpoints/hhi/Comp_v6_KLD01/opt.txt",
        "evaluator_checkpoint": ROOT / "checkpoints/hhi/text_mot_match/model/finest.tar",
        "evaluator_wrapper": ROOT / "models/evaluator_wrapper.py",
        "evaluator_option_loader": ROOT / "options/get_eval_option.py",
    }
    write_json(ordinary_audit, {
        "schema": "hhc_final_interx_ordinary_only_metric_audit_v1",
        "status": "pass",
        "metric_protocol": "official_interx_full_672d_ordinary_only",
        "evaluator_runtime": {"device_type": "cpu", "requested_device": "cpu"},
        "implementation": {
            key: artifact(path) for key, path in implementations.items()
        },
        "methods": {
            label: {
                "directory": str((fair_root / label).resolve()),
                "replications": 20,
                "summary": artifact(fair_root / label / "summary.json"),
                "records": [
                    {
                        "replication": replication,
                        "metrics": artifact(
                            fair_root / label / f"replication_{replication:02d}.json"
                        ),
                        "evaluator_provenance_verified": True,
                        "evaluator_runtime_verified": True,
                    }
                    for replication in range(20)
                ],
            }
            for label in LABELS
        },
    })

    paired_main = tmp_path / "paired_main.json"
    paired_timing = tmp_path / "paired_timing.json"
    write_json(paired_main, paired_payload(
        "hhc_no_timing", LABELS[0], {
            "intermask": LABELS[2], "interact2ar": LABELS[3],
            "coshmdm": LABELS[4], "timotion": LABELS[5],
        }, fair_root,
    ))
    write_json(paired_timing, paired_payload(
        "hhc_oracle", LABELS[1], {"hhc_no_timing": LABELS[0]}, fair_root,
    ))
    output_json = tmp_path / "table.json"
    output_markdown = tmp_path / "table.md"
    subprocess.run([
        sys.executable, str(ROOT / "tools/build_current_interx_main_table.py"),
        "--fair-eval-root", str(fair_root),
        "--paired-main", str(paired_main),
        "--paired-timing", str(paired_timing),
        "--ordinary-only-audit", str(ordinary_audit),
        "--output-json", str(output_json),
        "--output-markdown", str(output_markdown),
    ], check=True)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert set(payload["methods"]) == set(LABELS)
    assert payload["protocol"]["replications"] == 20
    assert "HHC (no contact timing)" in output_markdown.read_text(encoding="utf-8")
