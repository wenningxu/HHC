"""CPU self-check for the two-estimator volume quality gate."""
import json
import math
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_shape_gt_quality_gate import digest, fixture


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True))


def load_gate():
    fake = types.ModuleType("tools.run_interx_shape_augmentation")
    fake.sha256_file, fake.atomic_json = digest, write
    sys.modules[fake.__name__] = fake
    import importlib.util
    spec = importlib.util.spec_from_file_location("volume_gate", Path(__file__).with_name("build_shape_volume_quality_gate.py"))
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


def make_volume(root, report, reference_scores=None, candidate_scores=None):
    reference_scores = [1.0, 2.0] if reference_scores is None else reference_scores
    candidate_scores = [0.5, 1.5] if candidate_scores is None else candidate_scores
    gt, candidate = root / "gt", root / "candidate_volume"
    contract = {"backend": "cpu-fixture", "version": 1}
    for directory, values in ((gt, reference_scores), (candidate, candidate_scores)):
        (directory / "samples").mkdir(parents=True)
        records = []
        for row, value in zip(report["rows"], values if isinstance(values, list) else [values] * 2):
            p = directory / "samples" / f"{row['name']}.json"
            score = {"sample_key": row["name"], "seed": 7, "scoring_contract_sha256": "", "input_p1": "", "input_p2": "", "input_sha256": {}, "metrics": {"frames": row["frames"], "mean_foot_sliding_m_per_frame": value, "paper_eq6_squared_sum_m2_per_frame": value}}
            records.append(p)
            score['schema'] = 'hhc_volumetric_smpl_physical_metrics_v1'
            write(p, score)
        manifest = {"status": "complete", "sample_count": 2, "sample_records": [str(p.resolve()) for p in records], "scoring_contract": contract, "scoring_contract_sha256": __import__("tools.score_volumetric_smpl_npz", fromlist=["canonical_sha256"]).canonical_sha256(contract)}
        manifest['schema'] = 'hhc_volumetric_smpl_physical_metrics_v1'
        write(directory / "metrics.json", manifest)
        for p in records:
            score = json.loads(p.read_text()); score["scoring_contract_sha256"] = manifest["scoring_contract_sha256"]
            for person in (1, 2):
                src = (root / "candidate" / f"{p.stem}_p{person}.npz") if directory == candidate else (root / f"original_{p.stem}_p{person}.npz")
                if directory == gt: src.write_bytes(f"gt-{p.stem}-{person}".encode())
                score[f"input_p{person}"] = str(src.resolve()); score["input_sha256"][f"p{person}"] = digest(src)
            write(p, score)
    refs = {"candidate_manifest_sha256": digest(root / "candidate" / "_completion_manifest.json"), "source_motion_sha256": report["source_motion_sha256"], "records": []}
    for row in report["rows"]:
        b = row["input_bindings"]["cond"]
        refs["records"].append({"name": row["name"], "sidecar_sha256": b["sidecar_sha256"], "frames": row["frames"], "outputs": {f"original_gt_p{i}": {"path": str((root / f"original_{row['name']}_p{i}.npz").resolve()), "sha256": digest(root / f"original_{row['name']}_p{i}.npz")} for i in (1, 2)}})
    refpath = root / "references.json"; write(refpath, refs)
    return gt, candidate, refpath


def check():
    # This self-check only exercises the JSON gate path; avoid optional HDF5 imports.
    sys.modules.setdefault("h5py", types.ModuleType("h5py"))
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    score = types.ModuleType("tools.score_volumetric_smpl_npz")
    score.SCHEMA = 'hhc_volumetric_smpl_physical_metrics_v1'
    score.canonical_sha256 = lambda payload: __import__('hashlib').sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    sys.modules[score.__name__] = score
    from utils.interx_shape_augmentation import validated_quality_selection
    gate = load_gate()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); report_path, report = fixture(root)
        gt, candidate, refs = make_volume(root, report)
        # Proxy accepts only a; volume accepts both, but final acceptance stays a.
        result = gate.build_volume_gate(report_path, "cond", refs, gt, candidate)
        assert result["denominator"] == 2 and result["proxy_accepted_count"] == 1
        assert result["volume_accepted_count"] == 2 and result["accepted_names"] == ["a"]

        completion_path = root / "candidate" / "_completion_manifest.json"
        completion = json.loads(completion_path.read_text())
        gate_path = root / "quality_gate.json"
        write(gate_path, result)
        # The persisted gate is intentionally hashed separately from its payload.
        gate_entry = {"path": str(gate_path.resolve()), "sha256": digest(gate_path)}
        completions = [(completion_path, completion)]
        selected = validated_quality_selection([gate_entry], completions, ["b", "a"])
        assert selected["requested_count"] == 2
        assert selected["accepted_names"] == ["a"] and selected["rejected_names"] == ["b"]
        assert selected["pass_fraction"] == 0.5

        def rejects_selection(change):
            with tempfile.TemporaryDirectory() as bad_td:
                bad_root = Path(bad_td); bad_report_path, bad_report = fixture(bad_root)
                bad_gt, bad_candidate, bad_refs = make_volume(bad_root, bad_report)
                bad_result = gate.build_volume_gate(bad_report_path, "cond", bad_refs, bad_gt, bad_candidate)
                bad_gate_path = bad_root / "quality_gate.json"
                write(bad_gate_path, bad_result)
                bad_entry = {"path": str(bad_gate_path.resolve()), "sha256": digest(bad_gate_path)}
                bad_completion_path = bad_root / "candidate" / "_completion_manifest.json"
                bad_completion = json.loads(bad_completion_path.read_text())
                bad_completions = [(bad_completion_path, bad_completion)]
                bad_entries = [bad_entry]
                bad_requested = ["a", "b"]
                change(bad_root, bad_entries, bad_completions, bad_requested, bad_gate_path)
                try:
                    validated_quality_selection(bad_entries, bad_completions, bad_requested)
                except (KeyError, FileNotFoundError, ValueError, TypeError, AssertionError):
                    return
            raise AssertionError("invalid quality selection was accepted")

        rejects_selection(lambda r, e, c, n, p: e[0].update({"sha256": "bad"}))
        rejects_selection(lambda r, e, c, n, p: e.append(dict(e[0])))
        rejects_selection(lambda r, e, c, n, p: n.append("c"))
        def alter_decision(_, entries, __, ___, path):
            payload = json.loads(path.read_text()); payload["accepted_names"] = ["a", "b"]
            write(path, payload); entries[0]["sha256"] = digest(path)
        rejects_selection(alter_decision)

        def rejects(change):
            with tempfile.TemporaryDirectory() as bad_td:
                bad_root = Path(bad_td); bad_report_path, bad_report = fixture(bad_root)
                bad_gt, bad_candidate, bad_refs = make_volume(bad_root, bad_report)
                assert gate.build_volume_gate(bad_report_path, "cond", bad_refs, bad_gt, bad_candidate)['accepted_names'] == ['a']
                change(bad_root, bad_report, bad_refs, bad_gt, bad_candidate)
                try: gate.build_volume_gate(bad_report_path, "cond", bad_refs, bad_gt, bad_candidate)
                except (KeyError, FileNotFoundError, ValueError, TypeError, AssertionError): return
            raise AssertionError("invalid volume fixture was accepted")

        rejects(lambda r, rep, ref, g, c: ref.write_text(ref.read_text().replace('"frames": 2', '"frames": 3', 1)))
        rejects(lambda r, rep, ref, g, c: json.dump({**json.loads((g/'metrics.json').read_text()), "scoring_contract_sha256": "bad"}, (g/'metrics.json').open('w')))
        rejects(lambda r, rep, ref, g, c: json.dump({**json.loads((c/'samples/a.json').read_text()), "seed": 8}, (c/'samples/a.json').open('w')))
        rejects(lambda r, rep, ref, g, c: json.dump({**json.loads((c/'samples/a.json').read_text()), "metrics": {**json.loads((c/'samples/a.json').read_text())["metrics"], "mean_foot_sliding_m_per_frame": math.nan}}, (c/'samples/a.json').open('w')))
        rejects(lambda r, rep, ref, g, c: (g/'samples'/'b.json').unlink())
        rejects(lambda r, rep, ref, g, c: write(c/'samples/a.json', {**json.loads((c/'samples/a.json').read_text()), 'metrics': {'frames': 3}}))
        rejects(lambda r, rep, ref, g, c: write(g/'metrics.json', {**json.loads((g/'metrics.json').read_text()), 'schema': 'wrong'}))
        rejects(lambda r, rep, ref, g, c: write(c/'samples/a.json', {**json.loads((c/'samples/a.json').read_text()), 'schema': 'wrong'}))
        rejects(lambda r, rep, ref, g, c: (r/'original_a_p1.npz').write_bytes(b"changed-gt"))
        rejects(lambda r, rep, ref, g, c: (r/'candidate'/'a_p1.npz').write_bytes(b"changed-candidate"))
        rejects(lambda r, rep, ref, g, c: ref.write_text(ref.read_text().replace(rep["source_motion_sha256"], "bad-source")))
        rejects(lambda r, rep, ref, g, c: ref.write_text(ref.read_text().replace(digest(r/'candidate'/'_completion_manifest.json'), "bad-candidate")))
    print("shape volume quality gate checks: PASS")


if __name__ == "__main__": check()
