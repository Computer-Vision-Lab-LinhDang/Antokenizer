import tempfile
import unittest
from pathlib import Path

import torch

from mavt4d.object_state.v5.natural import (
    observation_schedule, mixture_counts, validate_clip, sample_plan,
)


class NaturalContracts(unittest.TestCase):
    def test_full_batch_quota_and_no_missing_source(self):
        self.assertEqual(mixture_counts(256), {"openimages": 51, "openvid": 64, "nuscenes": 141})
        self.assertEqual(sum(mixture_counts(257).values()), 257)

    def test_gap_has_history_and_revisit(self):
        for seed in range(10):
            observed = observation_schedule("G", seed)
            self.assertTrue(observed[0] and observed[1] and observed[-1])
            self.assertGreaterEqual(observed.count(False), 2)
            self.assertLessEqual(sum(observed), 6)

    def test_future_indices_never_input(self):
        plan = sample_plan(torch.arange(60).double() * .5, "G", seed=7, horizon=6)
        self.assertLess(max(plan["prefix"]), min(plan["future"]))
        self.assertEqual(len(set(plan["future"])), len(plan["future"]))
        self.assertGreaterEqual(len(plan["future"]), 1)

    def test_no_fake_temporal_target_from_single_image(self):
        plan = sample_plan(torch.tensor([0.], dtype=torch.float64), "S", seed=1, horizon=6)
        self.assertEqual(plan["prefix"], [0])
        self.assertEqual(plan["future"], [])

    def test_future_horizon_masks_unavailable_time(self):
        times = torch.arange(16).double() * .4
        plan = sample_plan(times, "S", seed=1, horizon=6)
        for frame in plan["future"]:
            offset = float(times[frame] - times[plan["prefix"][-1]])
            self.assertLessEqual(min(abs(offset - h * .5) for h in range(1, 7)), .05)

    def test_mixture_is_configurable_without_reassigning_missing_weight(self):
        self.assertEqual(mixture_counts(256, {"openimages": .5, "openvid": .5}),
                         {"openimages": 128, "openvid": 128})
        with self.assertRaises(ValueError):
            mixture_counts(256, {"openimages": .2, "openvid": .25})

    def test_clip_checks_actual_timestamps(self):
        clip = {"rgb": torch.zeros(2, 1, 3, 16, 16), "times": torch.tensor([0., 0.]),
                "rgb_valid": torch.ones(2, 1, 1, 16, 16, dtype=torch.bool)}
        with self.assertRaisesRegex(ValueError, "timestamps"):
            validate_clip(clip)


if __name__ == "__main__":
    unittest.main()
