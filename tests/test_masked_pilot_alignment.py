import unittest
from tools.compare_hhc_no_shape_masked_only_pilot import validation_values


class AlignmentTest(unittest.TestCase):
    def test_missing_event_is_not_shifted(self):
        rows = [{"value": 0.3, "preceding_training_optimizer_step": 20000},
                {"value": 0.2, "preceding_training_optimizer_step": 30000}]
        audit = {"metrics": {"MM": {"events": rows}}}
        self.assertEqual(validation_values(audit, "MM", 50000), {20000: 0.3, 30000: 0.2})
        rows.append(rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validation_values(audit, "MM", 50000)
        rows[:] = [{"value": 0.3, "wandb_step": 400}]
        with self.assertRaisesRegex(ValueError, "alignment"):
            validation_values(audit, "MM", 50000)


if __name__ == "__main__":
    unittest.main()
