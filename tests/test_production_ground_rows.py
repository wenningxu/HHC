import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.diagnose_shape_support_transfer import production_ground_rows
from tools.run_interx_shape_augmentation import sha256_file


SRC = Path("output/ssp62_production_physical_smoke2_alltiming_20260912")


class ProductionGroundRowsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "run"
        shutil.copytree(SRC, self.tmp)
        self.manifest = json.loads((self.tmp / "_completion_manifest.json").read_text())
        self.names = self.manifest["completed_names"]
        self.candidate = Path('output/ssp3d_all62_full6238_20260911/shard0')
        self.reference = json.loads((self.candidate / '_completion_manifest.json').read_text())
        floor_report = json.loads(Path('output/ssp62_mesh_ground_roundtrip_first100_20260912.json').read_text())
        self.floors = {(r['name'], r['condition']): r for r in floor_report['rows']}

    def tearDown(self):
        shutil.rmtree(self.tmp.parent)

    def test_valid_rows_bind_origin_and_hashes(self):
        rows, bindings = production_ground_rows(self.tmp, self.reference, self.candidate, self.floors)
        self.assertEqual(len(rows), 2)
        side = json.loads((self.tmp / f'{self.names[0]}.json').read_text())
        self.assertEqual(rows[0]["common_export_origin"], side['diagnostics']['ground_alignment']['common_export_origin'])
        self.assertIn(str(self.tmp / "_completion_manifest.json"), bindings)
        self.assertEqual(rows[0]["output_npz_sha256"], [sha256_file(self.tmp / f'{self.names[0]}_p{i}.npz') for i in (1, 2)])

    def test_rejects_tampered_provenance(self):
        cases = [("_completion_manifest.json", "complete", False),
                 ("_run_contract.json", "contract_sha256", "bad"),
                 (self.names[0] + ".json", "output_p1_sha256", "bad"),
                 (self.names[0] + ".json", "diagnostics", None)]
        for filename, key, value in cases:
            with self.subTest(key=key):
                data = json.loads((self.tmp / filename).read_text())
                if key == "diagnostics":
                    data[key]["ground_alignment"]["original_mesh_floors_m"][0] += 1
                else:
                    data[key] = value
                (self.tmp / filename).write_text(json.dumps(data))
                with self.assertRaises((AssertionError, ValueError, KeyError)):
                    production_ground_rows(self.tmp, self.reference, self.candidate, self.floors)
                shutil.copy2(SRC / filename, self.tmp / filename)


if __name__ == "__main__":
    unittest.main()
