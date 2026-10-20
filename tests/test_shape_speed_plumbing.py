import unittest
from unittest.mock import patch

import numpy as np
import torch
from types import SimpleNamespace

import eval.calc_pene_from_npz as pene
import optimize.optimize_full_body as opt


class StopAfterLoss(Exception):
    pass


def params():
    return {k: np.zeros(shape, np.float32) for k, shape in {
        "root_orient": (2, 3), "trans": (2, 3), "pose_body": (2, 21, 3),
        "pose_lhand": (2, 15, 3), "pose_rhand": (2, 15, 3), "betas": (10,),
    }.items()} | {"gender": "male"}


class ShapeSpeedPlumbingTest(unittest.TestCase):
    def test_static_models_forwarded(self):
        seen = []

        def loss(**kwargs):
            seen.append(kwargs["static_body_models"])
            raise StopAfterLoss

        for shared in (None, {"neutral": object(), "male": object(), "female": object()}):
            with patch.object(opt, "DyadicInteractionLoss", side_effect=loss):
                with self.assertRaises(StopAfterLoss):
                    opt.optimize_full_body(params(), params(), device="cpu", static_body_models=shared)
            self.assertIs(seen[-1], shared)

    def test_geometry_can_skip_normals(self):
        calc = pene.PeneCalculator.__new__(pene.PeneCalculator)
        verts = torch.ones(2, 3, 3)
        joints = torch.zeros(2, 2, 3)
        class FakeBM:
            f = "faces"
            def __call__(self, **kwargs):
                return SimpleNamespace(v=verts, Jtr=joints)
        calc.male_bm = FakeBM()
        p = {k: torch.as_tensor(v) for k, v in params().items() if k != "gender"}
        p["gender"] = "male"
        with patch.object(pene, "vertex_normals") as normals:
            a = calc._compute_geometry(p, compute_normals=False)
            normals.assert_not_called()
            b = calc._compute_geometry(p)
            normals.assert_called_once()
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertIsNone(a[2])


if __name__ == "__main__":
    unittest.main()
