import unittest
import torch

from mavt4d.object_state.v5.training import masked_mean, readout_objective, learning_rate


class NaturalTrainingLoss(unittest.TestCase):
    def test_missing_geometry_is_not_zero_depth_target(self):
        rgb = torch.full((1, 3, 3), .5, requires_grad=True)
        prediction = {"rgb": rgb, "range": torch.full((1, 3), 12., requires_grad=True),
                      "status_logits": torch.zeros(1, 3, 3, requires_grad=True)}
        targets = {"rgb": torch.ones(1, 3, 3), "rgb_valid": torch.ones(1, 3, dtype=torch.bool),
                   "range": torch.zeros(1, 3), "depth_valid": torch.zeros(1, 3, dtype=torch.bool),
                   "status": torch.full((1, 3), -1)}
        total, metrics = readout_objective(prediction, targets)
        self.assertAlmostEqual(total.item(), .5)
        self.assertEqual(metrics["depth"].item(), 0)
        self.assertEqual(metrics["status"].item(), 0)
        total.backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())

    def test_certified_unknown_is_masked_but_missing_is_not(self):
        pred = {"rgb": torch.zeros(1, 1, 3, requires_grad=True)}
        target = {"rgb": torch.ones(1, 1, 3), "rgb_valid": torch.ones(1, 1,dtype=torch.bool),
                  "status": torch.full((1, 1), 2)}
        loss, _ = readout_objective(pred, target)
        self.assertEqual(loss.item(), 0)
        target["status"].fill_(-1)
        loss, _ = readout_objective(pred, target)
        self.assertEqual(loss.item(), 1)

    def test_empty_mask_differentiable_zero(self):
        x = torch.ones(2, requires_grad=True)
        result = masked_mean(x, torch.zeros(2, dtype=torch.bool))
        result.backward()
        self.assertEqual(result.item(), 0)
        self.assertEqual(x.grad.sum().item(), 0)

    def test_lr_preserves_full_budget(self):
        self.assertAlmostEqual(learning_rate(999, 20000), 3e-4)
        self.assertAlmostEqual(learning_rate(19999, 20000), 3e-5)

    def test_real_model_two_observations_rollout_backward(self):
        from mavt4d.object_state.v5.config import V5Config
        from mavt4d.object_state.v5.model import V5Model
        from mavt4d.object_state.v5.natural import sample_plan
        from mavt4d.object_state.v5.training import compute_loss
        torch.manual_seed(17)
        model = V5Model(V5Config(slots=2, content_tokens=2, dynamic_tokens=2,
            width=16, evidence_width=24, spatial_blocks=2, spatial_heads=3,
            resampler_heads=4, samples_per_ray=4, query_chunk_size=4,
            window_size=2, relay_tokens=2))
        times = torch.arange(14).double() * .5
        clip = {"rgb": torch.rand(14, 1, 3, 32, 32), "times": times,
                "rgb_valid": torch.ones(14, 1, 1, 32, 32, dtype=torch.bool),
                "source": "openvid", "plan": sample_plan(times, "S", horizon=2)}
        loss, metrics = compute_loss(model, [clip, clip], step=0, rays=4)
        loss.backward()
        self.assertEqual(metrics["microbatch"], 2)
        self.assertEqual(metrics["future_valid"], 2)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
