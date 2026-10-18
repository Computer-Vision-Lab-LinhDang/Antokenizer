import unittest
import torch

from mavt4d.object_state.v5.evaluation import build_suite, validate_suite, ReadoutMetrics, copy_last, horizon_seconds


class Dataset:
    split = "val"
    keys = {"dl3dv": ["a", "b", "c"]}
    manifest_hashes = {"dl3dv": "hash-a"}


class EvaluationTests(unittest.TestCase):
    def test_horizon_uses_time_not_available_target_rank(self):
        clip = {"times": torch.tensor([0., .5, 1.5]), "plan": {"prefix": [1], "future": [2]}}
        self.assertEqual(horizon_seconds(clip, 2), 1.)

    def test_suite_pins_groups_inventory_and_seed(self):
        a = build_suite(Dataset(), groups_per_source=2, seed=7)
        self.assertEqual(a, build_suite(Dataset(), groups_per_source=2, seed=7))
        self.assertEqual(len(a["groups"]["dl3dv"]), 2)
        validate_suite(a, Dataset())
        changed = Dataset()
        changed.manifest_hashes = {"dl3dv": "hash-b"}
        with self.assertRaises(ValueError):
            validate_suite(a, changed)

    def test_dev_cannot_be_academic_validation(self):
        data = Dataset(); data.split = "dev"
        with self.assertRaises(ValueError):
            build_suite(data)

    def test_metrics_distinguish_missing_and_unknown(self):
        metrics = ReadoutMetrics()
        target = {"rgb": torch.ones(1, 3, 3), "rgb_valid": torch.ones(1, 3, dtype=torch.bool),
                  "status": torch.tensor([[0, 2, -1]])}
        metrics.add({"rgb": torch.zeros(1, 3, 3)}, target)
        result = metrics.result()
        self.assertEqual(result["rgb_rays"], 2)
        self.assertEqual(result["rgb_mae"], 1)
        self.assertAlmostEqual(result["status_label_coverage"], 2 / 3)

    def test_copy_last_uses_only_last_allowed_frame_same_view(self):
        clip = {"rgb": torch.arange(5.).view(5, 1, 1, 1, 1).expand(5, 1, 3, 2, 2),
                "rgb_valid": torch.ones(5, 1, 1, 2, 2, dtype=torch.bool),
                "plan": {"prefix": [1, 2, 3], "observed": [True, False, True]}}
        target = {"pixels": torch.tensor([[[.5, .5]]]), "view": 0,
                  "rgb": torch.zeros(1, 1, 3), "rgb_valid": torch.ones(1, 1,dtype=torch.bool)}
        pred, _ = copy_last(clip, target, frame=2)
        self.assertEqual(pred["rgb"][0, 0, 0], 1)
        pred, _ = copy_last(clip, target, frame=4)
        self.assertEqual(pred["rgb"][0, 0, 0], 3)


if __name__ == "__main__":
    unittest.main()
