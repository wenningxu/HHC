"""CPU self-check for shape-GT gate input binding and decision integrity."""
import copy
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tools" / "build_shape_gt_quality_gate.py"
METRICS = ("foot_sliding_m_per_frame", "nearest_vertex_squared_sum_m2_per_frame")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def fixture(root):
    folder = root / "candidate"
    folder.mkdir()
    source = root / "source.motion"
    source.write_bytes(b"minimal-source")
    names = ["a", "b"]
    bindings = {}
    for name in names:
        sidecar = folder / f"{name}.json"
        bindings[name] = {}
        for person in (1, 2):
            motion = folder / f"{name}_p{person}.npz"
            motion.write_bytes(f"{name}-{person}".encode())
            bindings[name][f"p{person}_sha256"] = digest(motion)
        sidecar.write_text(json.dumps({"crop_start": 0, "crop_stop": 2,
                                       "output_p1_sha256": bindings[name]["p1_sha256"],
                                       "output_p2_sha256": bindings[name]["p2_sha256"]}))
        bindings[name]["sidecar_sha256"] = digest(sidecar)
        bindings[name]["output_p1_sha256"] = bindings[name]["p1_sha256"]
        bindings[name]["output_p2_sha256"] = bindings[name]["p2_sha256"]
    completion = {
        "complete": True, "completed_names": names, "completed_samples": 2,
        "requested_samples": 2, "source_motion": str(source),
        "source_motion_sha256": digest(source),
    }
    completion_path = folder / "_completion_manifest.json"
    write_json(completion_path, completion)
    rows = []
    for name, passed in (("a", True), ("b", False)):
        gt = {key: 0.0 for key in METRICS}
        candidate = {key: 0.0 if passed else 0.1 for key in METRICS}
        rows.append({"name": name, "frames": 2, "input_bindings": {"cond": bindings[name]},
                     "conditions": {"original_gt": gt, "cond": candidate},
                     "proxy_pass_vs_original_gt": {"cond": passed}})
    report = {
        "status": "diagnostic_only_nearest_vertex_proxy", "condition_directories": {"cond": str(folder)},
        "condition_completion_sha256": {"cond": digest(completion_path)},
        "source_motion": str(source), "source_motion_sha256": digest(source), "rows": rows,
    }
    report_path = root / "report.json"
    write_json(report_path, report)
    return report_path, report


def run(report):
    # The production module imports GPU/data dependencies; this gate itself only
    # needs two tiny helpers, so keep this check CPU-only and import the real gate.
    module = types.ModuleType("tools.run_interx_shape_augmentation")
    module.sha256_file = digest
    module.atomic_json = write_json
    sys.modules[module.__name__] = module
    import importlib.util
    spec = importlib.util.spec_from_file_location("shape_gate", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate.build_gate(report, "cond")


def check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report_path, report = fixture(root)
        decision = run(report_path)
        assert decision["denominator"] == 2 and decision["accepted_count"] == 1
        assert decision["rejected_names"] == ["b"]  # GT=0 rejects positive candidate.

        def rejects(mutator):
            mutated = copy.deepcopy(report)
            mutator(mutated, root)
            write_json(report_path, mutated)
            try:
                run(report_path)
            except (KeyError, ValueError, TypeError):
                pass
            else:
                raise AssertionError("invalid fixture was accepted")
            write_json(report_path, report)

        rejects(lambda r, _: r["rows"][0].update({"input_bindings": {}}))
        rejects(lambda r, _: r["rows"].pop())
        rejects(lambda r, _: r["rows"].append(copy.deepcopy(r["rows"][0])))
        def bad_denominator(r, root):
            path = root / "candidate" / "_completion_manifest.json"
            completion = json.loads(path.read_text())
            completion["requested_samples"] = 3
            write_json(path, completion)
            r["condition_completion_sha256"]["cond"] = digest(path)
        rejects(bad_denominator)
        completion_path = root / "candidate" / "_completion_manifest.json"
        completion = json.loads(completion_path.read_text())
        completion["requested_samples"] = 2
        write_json(completion_path, completion)
        rejects(lambda r, _: r["rows"][0]["conditions"]["cond"].update({METRICS[0]: float("nan")}))
        rejects(lambda r, _: r["rows"][0]["proxy_pass_vs_original_gt"].update({"cond": False}))
        rejects(lambda r, _: r.pop("source_motion_sha256"))
        # Mutating immutable artifacts leaves their recorded hashes stale.
        (root / "candidate" / "a_p1.npz").write_bytes(b"changed")
        try:
            run(report_path)
        except ValueError:
            pass
        else:
            raise AssertionError("changed NPZ was accepted")
        (root / "candidate" / "a_p1.npz").write_bytes(b"a-1")
        (root / "candidate" / "a.json").write_text('{"crop_start": 0, "crop_stop": 3}')
        try:
            run(report_path)
        except ValueError:
            pass
        else:
            raise AssertionError("changed metadata was accepted")
    print("shape GT quality gate checks: PASS")


if __name__ == "__main__":
    check()
