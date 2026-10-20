import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.build_physical_shape_quality_gate
import tools.build_shape_volume_quality_gate
from utils.interx_shape_augmentation import validated_quality_selection


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QualitySelectionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.completion = root / "_completion_manifest.json"
        self.completion.write_text("{}")
        self.report = root / "report.json"
        self.report.write_text("{}")
        self.floor = root / "floor.json"
        self.floor.write_text("{}")
        self.candidate = root / "candidate"
        self.candidate.mkdir()
        self.completions = [(self.completion, {"completed_names": ["a", "b"]})]

    def tearDown(self):
        self.tmp.cleanup()

    def gate(self, schema, accepted=("a",)):
        gate_path = Path(self.tmp.name) / f"{schema}.json"
        gate = {
            "schema": schema,
            "completion": str(self.completion),
            "report": str(self.report),
            "report_sha256": _sha(self.report),
            "accepted_names": list(accepted),
        }
        if schema == "hhc_physical_mesh_ground_quality_gate_v1":
            gate.update(candidate_dir=str(self.candidate), floor_report=str(self.floor))
        else:
            gate.update(condition="original_shape", references=str(self.report),
                        gt_dir=str(self.candidate), candidate_dir=str(self.candidate))
        gate_path.write_text(json.dumps(gate))
        return [{"path": str(gate_path), "sha256": _sha(gate_path)}], gate

    def select(self, schema, accepted=("a",), requested=("a", "b")):
        entries, gate = self.gate(schema, accepted)
        builder = lambda *args: gate
        physical = patch("tools.build_physical_shape_quality_gate.build_physical_gate", builder)
        volume = patch("tools.build_shape_volume_quality_gate.build_volume_gate", builder)
        return entries, gate, physical, volume, requested

    def test_old_policy_is_preserved(self):
        entries, gate, physical, volume, requested = self.select(
            "hhc_original_gt_two_estimator_quality_gate_v1")
        with physical, volume as volume_mock:
            result = validated_quality_selection(entries, self.completions, requested)
        self.assertEqual(result["policy"], "original_gt_two_estimator_intersection_no_tolerance")

    def test_physical_policy_returns_correct_selection(self):
        entries, gate, physical, volume, requested = self.select(
            "hhc_physical_mesh_ground_quality_gate_v1")
        with physical as physical_mock, volume:
            result = validated_quality_selection(entries, self.completions, requested)
        self.assertEqual(result["policy"], "physical_original_gt_volumetric_no_tolerance")
        self.assertEqual(result["accepted_names"], ["a"])
        self.assertEqual(result["rejected_names"], ["b"])
        self.assertEqual(result["requested_count"], 2)

    def test_mixed_policies_are_rejected(self):
        old_entries, old_gate, physical, volume, requested = self.select(
            "hhc_original_gt_two_estimator_quality_gate_v1")
        second = Path(self.tmp.name) / "_completion_manifest_2.json"
        second.write_text("{}")
        new_gate = dict(old_gate, schema="hhc_physical_mesh_ground_quality_gate_v1",
                        completion=str(second), candidate_dir=str(self.candidate),
                        floor_report=str(self.floor))
        new_path = Path(self.tmp.name) / "physical.json"
        new_path.write_text(json.dumps(new_gate))
        new_entries = [{"path": str(new_path), "sha256": _sha(new_path)}]
        entries = old_entries + new_entries
        completions = self.completions + [(second, {"completed_names": []})]
        with patch("tools.build_physical_shape_quality_gate.build_physical_gate", lambda *args: new_gate), \
             patch("tools.build_shape_volume_quality_gate.build_volume_gate", lambda *args: old_gate) as volume_mock:
            with self.assertRaisesRegex(ValueError, "Cannot mix"):
                validated_quality_selection(entries, completions, requested)

    def test_partial_acceptance_keeps_full_requested_denominator(self):
        entries, _, physical, volume, requested = self.select(
            "hhc_physical_mesh_ground_quality_gate_v1", accepted=("a",))
        with physical, volume:
            result = validated_quality_selection(entries, self.completions, requested)
        self.assertEqual(result["requested_count"], 2)
        self.assertEqual(result["pass_fraction"], 0.5)

    def test_tampered_accepted_names_are_rejected(self):
        entries, gate, physical, volume, requested = self.select(
            "hhc_physical_mesh_ground_quality_gate_v1")
        tampered = dict(gate, accepted_names=["b"])
        gate_path = Path(entries[0]["path"])
        gate_path.write_text(json.dumps(tampered))
        entries[0]["sha256"] = _sha(gate_path)
        builder = lambda *args: gate
        with physical, volume:
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                with patch("tools.build_physical_shape_quality_gate.build_physical_gate", builder):
                    validated_quality_selection(
                        [{"path": entries[0]["path"], "sha256": entries[0]["sha256"]}],
                        self.completions, requested)

    def test_semantic_exclusions_preserve_denominator_and_physical_decisions(self):
        entries, _, physical, volume, requested = self.select(
            'hhc_physical_mesh_ground_quality_gate_v1', accepted=('a', 'b'))
        path = Path(self.tmp.name) / 'review.json'
        review = {'status': 'withhold_pending_semantic_review',
                  'entries': [{'augmented_name': 'a', 'reason': 'unresolved actor mismatch'}]}
        path.write_text(json.dumps(review))
        binding = {'path': str(path), 'sha256': _sha(path)}
        with physical, volume:
            result = validated_quality_selection(entries, self.completions, requested, binding)
            self.assertEqual(result['physical_accepted_names'], ['a', 'b'])
            self.assertEqual(result['accepted_names'], ['b'])
            self.assertEqual(result['requested_count'], 2)
            self.assertEqual(result['pass_fraction'], .5)
            self.assertEqual(result, validated_quality_selection(
                entries, self.completions, requested, result['semantic_exclusions']))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                validated_quality_selection(entries, self.completions, requested, binding)
            for bad in ([{'augmented_name': 'unknown', 'reason': 'risk'}],
                        review['entries'] * 2, [{'augmented_name': 'a', 'reason': ''}]):
                path.write_text(json.dumps(dict(review, entries=bad)))
                with self.assertRaises(ValueError):
                    validated_quality_selection(entries, self.completions, requested,
                                                {'path': str(path), 'sha256': _sha(path)})


if __name__ == "__main__":
    unittest.main()
