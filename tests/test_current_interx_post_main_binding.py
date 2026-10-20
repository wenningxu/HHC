import json
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_hhc_sampling_pareto import sha256_file, validate_headline_generation
from tools.build_current_shape_augmentation_results import (
    validate_artifact_record as validate_shape_artifact_record,
)
from tools.build_current_interx_review_bundle import (
    validate_embedded_artifacts,
    validate_multimodality_report,
)


NO_TIMING = "hhc_historical_plus_paper_mask_ratio_no_contact_common50"
ORACLE = "hhc_historical_plus_paper_mask_ratio_contact_common50"
OLD_NO_TIMING = "hhc_formal_raw12288_repaired_mgt_common50_no_contact"
OLD_ORACLE = "hhc_formal_raw12288_repaired_mgt_common50_contact"


def test_current_optimization_defaults_to_current_no_timing_hhc():
    source = (ROOT / "tools/run_paper_optimization_formal.py").read_text(
        encoding="utf-8"
    )
    assert NO_TIMING in source
    assert OLD_NO_TIMING not in source
    assert "hhc_historical_plus_paper_mask_ratio_no_contact_common50_rep0" in source
    assert "optimization_historical_paper_mask_rep0" in source
    assert '"contact_guidance": False' in source
    assert '"contact_timing_source": "none"' in source
    assert '"sampling_steps": 50' in source
    assert '"signed_distance_backend": "nearest_vertex"' in source
    assert '"--signed-distance-backend", "nearest_vertex"' in source

    optimizer = (ROOT / "tools/run_paper_contact_optimization.py").read_text(
        encoding="utf-8"
    )
    assert "default='nearest_vertex'" in optimizer
    assert "signed_distance_backend=args.signed_distance_backend" in optimizer

    ablation = (ROOT / "tools/run_paper_optimization_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "default='nearest_vertex'" in ablation
    assert "'--signed-distance-backend', args.signed_distance_backend" in ablation


def test_current_physical_and_qualitative_methods_are_exact():
    expected = {
        NO_TIMING,
        ORACLE,
        "intermask_official_exact_common50",
        "interact2ar_official_fair_no_gt_first_frame",
        "coshmdm_interx_retrained_2daycap_seed123",
        "timotion_transformer_lpa_interx_retrained_fair",
    }
    physical = (ROOT / "tools/run_final_interx_mesh_physical.py").read_text(
        encoding="utf-8"
    )
    prepare = (ROOT / "tools/prepare_qualitative_comparison_selection.py").read_text(
        encoding="utf-8"
    )
    assert NO_TIMING in physical and ORACLE in physical
    assert OLD_NO_TIMING not in physical and OLD_ORACLE not in physical
    assert NO_TIMING in prepare and ORACLE in prepare
    assert OLD_NO_TIMING not in prepare and OLD_ORACLE not in prepare
    selection = json.loads((
        ROOT / "configs/qualitative_comparison_selection.json"
    ).read_text(encoding="utf-8"))
    assert selection["status"] == "pass"
    assert set(selection["methods"]) == expected
    assert selection["selection_is_metric_independent"] is True
    assert len(selection["records"]) == 9


def test_current_review_driver_is_direct_and_shape_waits_for_it():
    driver = (
        ROOT / "tools/run_current_interx_review_experiments_before_shape.sh"
    ).read_text(encoding="utf-8")
    assert "while " not in driver
    assert "sleep " not in driver
    assert "systemctl" not in driver
    assert NO_TIMING in driver and ORACLE in driver
    assert OLD_NO_TIMING not in driver and OLD_ORACLE not in driver
    assert "output/current_interx_main_table.json" in driver
    assert "tools/run_paper_optimization_formal.py" in driver
    assert "tools/audit_current_interx_optimization_backend.py" in driver
    assert "tools/run_final_interx_mesh_physical.py" in driver
    assert "tools/render_fair_interx_qualitative.py" in driver
    assert "tools/run_hhc_contact_timing_ablation.py" in driver
    assert "tools/summarize_hhc_contact_timing_ablation.py" in driver
    assert "tools/run_hhc_sampling_pareto.py" in driver
    assert "tools/summarize_hhc_sampling_pareto.py" in driver
    required_outputs = (
        "output/optimization_historical_paper_mask_rep0/formal_manifest.json",
        "output/current_interx_optimization_backend_contract.json",
        "output/current_interx_mesh_physical/manifest.json",
        "output/current_interx_qualitative/manifest.json",
        (
            "output/fair_eval/"
            "hhc_historical_paper_mask_contact_timing_ablation_rep0/summary.json"
        ),
        (
            "output/fair_eval/"
            "hhc_historical_paper_mask_sampling_pareto_no_timing_rep0/summary.json"
        ),
    )
    for launcher in (
        "tools/run_interx_shape_augmentation_nearest_last_stage.sh",
        "tools/run_hhc_shape_aug_historical_paper_mask_arm.sh",
    ):
        source = (ROOT / launcher).read_text(encoding="utf-8")
        for output in required_outputs:
            assert output in source


def test_current_timing_ablation_uses_historical_model_and_assets():
    runner = (ROOT / "tools/run_hhc_contact_timing_ablation.py").read_text(
        encoding="utf-8"
    )
    summary = (ROOT / "tools/summarize_hhc_contact_timing_ablation.py").read_text(
        encoding="utf-8"
    )
    for source in (runner, summary):
        assert NO_TIMING in source and ORACLE in source
        assert OLD_NO_TIMING not in source and OLD_ORACLE not in source
    assert "hhc_historical_plus_paper_mask_ratio_selection.json" in runner
    assert "vq_precleanup_raw12288_historical_mmm_cache" in runner
    assert "'--mgt-contract', 'historical_mmm'" in runner
    assert "'--inference-conditioning-contract', 'paper_text_cfg'" in runner
    assert "'--contact-map-contract', 'paper_formula_train_only'" in runner


def test_current_sampling_pareto_uses_no_timing_historical_model():
    runner = (ROOT / "tools/run_hhc_sampling_pareto.py").read_text(
        encoding="utf-8"
    )
    assert "hhc_historical_plus_paper_mask_ratio_selection.json" in runner
    assert "hhc_historical_paper_mask_sampling_pareto_no_timing_rep0" in runner
    assert "'--mgt-contract', 'historical_mmm'" in runner
    assert "'--inference-conditioning-contract', 'paper_text_cfg'" in runner
    assert "'--no-contact-guidance'" in runner
    assert "'--contact-timing-source', 'none'" in runner
    assert "'--contact-guidance'" not in runner
    assert "reused_current_headline_generation" in runner


def test_sampling_pareto_headline_reuse_is_contract_bound():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        requests = root / "requests.npz"
        requests.write_bytes(b"frozen-request")
        generation = root / "main.h5"
        expected = {
            "requests_sha256": sha256_file(requests),
            "checkpoint_sha256": "checkpoint-hash",
            "vq_contract": "precleanup_raw12288",
            "mgt_contract": "historical_mmm",
            "inference_conditioning_contract": "paper_text_cfg",
            "sampler": "sample_intra_cas",
            "sampling_steps": 50,
            "temperature": 0.5,
            "cfg_scale": 3.5,
            "contact_guidance": False,
            "contact_timing_source": "none",
            "seed": 1234,
        }
        with h5py.File(generation, "w") as output:
            for key, value in expected.items():
                output.attrs[key] = value
            output.create_dataset(
                "completed_repeats", data=np.ones(1, dtype=np.bool_)
            )
        validate_headline_generation(
            generation, requests, "checkpoint-hash", 1234, 1
        )
        with h5py.File(generation, "r+") as output:
            output.attrs["sampling_steps"] = 10
        try:
            validate_headline_generation(
                generation, requests, "checkpoint-hash", 1234, 1
            )
        except ValueError as error:
            assert "another contract" in str(error)
        else:
            raise AssertionError("stale headline generation was reused")


def test_shape_final_report_rehashes_internal_contract_artifacts():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "launcher.sh"
        path.write_text("frozen\n", encoding="utf-8")
        record = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        assert validate_shape_artifact_record(record, "fixture") == path.resolve()
        path.write_text("changed\n", encoding="utf-8")
        try:
            validate_shape_artifact_record(record, "fixture")
        except ValueError as error:
            assert "stale" in str(error)
        else:
            raise AssertionError("stale shape contract artifact was accepted")


def test_current_review_bundle_recursively_rehashes_artifacts():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        path.write_text("result\n", encoding="utf-8")
        payload = {"nested": [{
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }]}
        assert validate_embedded_artifacts(payload) == 1
        path.write_text("tampered\n", encoding="utf-8")
        try:
            validate_embedded_artifacts(payload)
        except ValueError as error:
            assert "stale" in str(error)
        else:
            raise AssertionError("stale nested review artifact was accepted")


def test_current_shape_finalizer_uses_matched_current_arms():
    finalizer = (
        ROOT / "tools/finalize_current_interx_shape_augmentation_results.sh"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "tools/build_current_shape_augmentation_results.py"
    ).read_text(encoding="utf-8")
    assert "while " not in finalizer and "sleep " not in finalizer
    for condition in ("no_contact", "contact"):
        assert f"hhc_historical_plus_paper_mask_ratio_{condition}_common50" in finalizer
        for arm in ("random", "optimized"):
            label = f"hhc_historical_paper_mask_shapeaug_{arm}_{condition}_common50"
            assert label in finalizer and label in builder
    assert "paired_current_shape_augmentation_no_contact.json" in finalizer
    assert "paired_current_shape_augmentation_contact.json" in finalizer
    assert "current_interx_shape_augmentation_results.json" in finalizer
    assert "remote_state') != 'finished'" in builder
    assert "interx_shape_aug_random_seed123_v3" in builder
    assert "interx_shape_aug_optimized_nearest_vertex_seed123_v3" in builder
    assert "current_interx_optimization_backend_contract.json" in builder
    assert "validate_artifact_record" in builder
    assert "optimization_signed_distance_backend" in builder
    bundle = (ROOT / "tools/build_current_interx_review_bundle.py").read_text(
        encoding="utf-8"
    )
    assert "validate_embedded_artifacts" in bundle
    assert "hhc_current_interx_main_table_v1" in bundle
    assert "hhc_current_shape_augmentation_results_v1" in bundle
    assert '== "nearest_vertex"' in bundle
    assert "standard_row_concatenated_v1" in bundle
    assert '"all_pairs_per_prompt") == 435' in bundle
    assert '"multimodality_audit_summary.json"' in bundle
    assert 'official.get("values") == official_main.get("values")' in bundle
    assert '"exhaustive_all_435_pairs"' in bundle
    assert '"coshmdm_diagnosis"' in bundle
    assert "coshmdm_training_trajectory_diagnosis.json" in bundle
    assert "coshmdm_embedding_gap_diagnosis.json" in bundle
    assert "coshmdm_representation_adapter_audit_live.json" in bundle
    assert "implementation_source_stability_verified_at_completion" in bundle
    assert "coshmdm_interx_reproduction_diagnosis.md" in bundle
    assert '"not_a_body_controlled_fid": True' in bundle
    assert "tools/build_current_interx_review_bundle.py" in finalizer


def test_current_bundle_rejects_stale_exhaustive_mm_official_vector():
    values = [float(index) for index in range(20)]
    main_values = list(values)
    report = {
        "replications": 20,
        "all_pairs_per_prompt": 435,
        "legacy_provenance_allowed": False,
        "official_10_pairs": {"mean": 9.5, "ci95": 1.0, "values": values},
        "exhaustive_all_pairs": {
            "mean": 9.4, "ci95": 0.2, "values": [value + 0.1 for value in values],
        },
        "official_minus_all": {"mean": -0.1},
        "root_mean_square_pair_subsampling_error": 0.5,
        "per_replication": [
            {"replication": index} for index in range(20)
        ],
    }
    compact = validate_multimodality_report(
        report, {"values": main_values}, "fixture"
    )
    assert compact["exhaustive_all_435_pairs"]["mean"] == 9.4
    report["official_10_pairs"]["values"][3] += 1e-7
    try:
        validate_multimodality_report(report, {"values": main_values}, "fixture")
    except ValueError as error:
        assert "official values are stale" in str(error)
    else:
        raise AssertionError("stale exhaustive-MM official vector was accepted")


if __name__ == "__main__":
    test_current_optimization_defaults_to_current_no_timing_hhc()
    test_current_physical_and_qualitative_methods_are_exact()
    test_current_review_driver_is_direct_and_shape_waits_for_it()
    test_current_timing_ablation_uses_historical_model_and_assets()
    test_current_sampling_pareto_uses_no_timing_historical_model()
    test_sampling_pareto_headline_reuse_is_contract_bound()
    test_shape_final_report_rehashes_internal_contract_artifacts()
    test_current_review_bundle_recursively_rehashes_artifacts()
    test_current_shape_finalizer_uses_matched_current_arms()
    test_current_bundle_rejects_stale_exhaustive_mm_official_vector()
    print("current Inter-X post-main bindings passed")
