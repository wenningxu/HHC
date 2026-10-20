from types import SimpleNamespace
import unittest
import torch
from utils.losses import ReConsLossInter, ReConsLossSingle


class RotationLossAutocastTest(unittest.TestCase):
    def test_endpoint_and_nearby_gradients_match_fp32(self):
        methods = [(ReConsLossInter.forward_rotation, 2),
                   (ReConsLossInter.forward_relative_oriental, 2),
                   (ReConsLossSingle.forward_rotation, 1)]
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            for method, actors in methods:
                for offset in (0., .01):
                    base = torch.tensor([1., 0., 0., 0., 1., 0.], device=device).repeat(1, 2, 56, actors)
                    results = []
                    for amp in (False, True):
                        pred = (base + offset).requires_grad_()
                        obj = SimpleNamespace(motion_pred_de=pred, motion_gt_de=base)
                        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16, enabled=amp):
                            loss = method(obj)
                        loss.backward()
                        self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(pred.grad).all())
                        results.append((loss.detach(), pred.grad))
                    for reference, actual in zip(*results):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
