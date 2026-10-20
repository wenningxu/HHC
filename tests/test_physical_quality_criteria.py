import copy
import unittest

from tools.build_physical_shape_quality_gate import passes_quality


def sidecar(changed=True):
    return {"crop_start": 3, "crop_stop": 5,
            "source_shapes": {"betas": [[0.0], [1.0]]},
            "augmented_shapes": [{"betas": [2.0 if changed else 0.0]},
                                  {"betas": [3.0 if changed else 1.0]}]}


def score(value=1.0, frames=2, observations=1):
    return {"frames": frames, "mean_foot_sliding_m_per_frame": value,
            "paper_eq6_squared_sum_m2_per_frame": value,
            "foot_sliding_p1": {"contact_observations": observations},
            "foot_sliding_p2": {"contact_observations": 0}}


class PhysicalQualityCriteriaTest(unittest.TestCase):
    def test_equal_metrics_and_both_changed_shapes_pass(self):
        self.assertTrue(passes_quality({"original_shape": score(), "optimized": score()}, sidecar()))

    def test_any_metric_regression_rejects(self):
        for metric in ('paper_eq6_squared_sum_m2_per_frame', 'mean_foot_sliding_m_per_frame'):
            v = {"original_shape": score(), "optimized": score()}
            v["optimized"][metric] += 1e-12
            self.assertFalse(passes_quality(v, sidecar()))

    def test_shape_observability_and_validation(self):
        self.assertFalse(passes_quality({"original_shape": score(), "optimized": score()}, sidecar(False)))
        for i in (0, 1):
            s = sidecar()
            s['augmented_shapes'][i]['betas'] = s['source_shapes']['betas'][i]
            self.assertFalse(passes_quality({'original_shape': score(), 'optimized': score()}, s))
        for condition in ('original_shape', 'optimized'):
            v = {'original_shape': score(), 'optimized': score()}
            v[condition] = score(observations=0)
            self.assertFalse(passes_quality(v, sidecar()))
        for bad in (float("nan"), -1.0):
            with self.assertRaises(ValueError):
                passes_quality({"original_shape": score(bad), "optimized": score()}, sidecar())
        with self.assertRaises(ValueError):
            passes_quality({"original_shape": score(frames=3), "optimized": score()}, sidecar())


if __name__ == "__main__":
    unittest.main()
