import tempfile
from pathlib import Path
import unittest
import torch

from mavt4d.object_state.v5.runner import _save, _rng, _restore_rng, validate_run_config


class RunnerTests(unittest.TestCase):
    def test_full_budget_and_exact_mixture_are_required(self):
        config = {"schema": "natural-v5-run-v1", "training": {"updates": 20000, "effective_batch": 256},
                  "mixture": {"dl3dv": 1.}, "data": {"dl3dv_manifest": "manifest.jsonl"}}
        validate_run_config(config)
        config["data"]["mixture"] = {"co3d": 1.}
        with self.assertRaises(ValueError): validate_run_config(config)
        config["data"]["mixture"] = {"dl3dv": 1.}
        config["training"]["updates"] = 20
        with self.assertRaises(ValueError): validate_run_config(config)

    def test_checkpoint_restores_optimizer_and_rng_boundary(self):
        torch.manual_seed(77)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1,3)).sum().backward(); optimizer.step(); optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"last.pt"
            _save(path, model, optimizer, lock={"test": True}, step=1, run_id=None, capacities={"dl3dv":1})
            expected = torch.rand(4)
            saved = torch.load(path, weights_only=False)
            model2 = torch.nn.Linear(3,2)
            optimizer2 = torch.optim.AdamW(model2.parameters())
            model2.load_state_dict(saved["model"]); optimizer2.load_state_dict(saved["optimizer"])
            _restore_rng(saved["rng"])
            self.assertTrue(torch.equal(expected, torch.rand(4)))
            self.assertEqual(saved["step"], 1)
            self.assertTrue(torch.equal(model.weight, model2.weight))
            self.assertEqual(len(optimizer.state), len(optimizer2.state))


class CapacityTests(unittest.TestCase):
    def _probe(self, *, seconds=900, stop_after=None, deadline=None, nonfinite=None, prefilled=False):
        import copy
        from unittest.mock import patch, Mock
        from types import SimpleNamespace
        from mavt4d.object_state.v5 import runner
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        if prefilled:
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        original_optimizer = copy.deepcopy(optimizer.state_dict())
        original = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        clock, calls = [0.], []
        dataset = Mock()
        dataset.draw.side_effect = lambda source, *a, **kw: {"source": source, "plan": {"profile": "G", "future": list(range(6))}}
        def loss_fn(model, clips, **kwargs):
            calls.append((clips[0]["source"], len(clips)))
            clock[0] += 1.
            torch.rand(2)
            value = model(torch.ones(1, 1)).sum()
            if nonfinite == "loss":
                value = value * float("nan")
            elif nonfinite == "gradient":
                value.register_hook(lambda grad: grad * float("nan"))
            return value, {}
        try:
            with patch.object(runner, "compute_loss", side_effect=loss_fn), \
                 patch.object(runner.time, "monotonic", side_effect=lambda: clock[0]), \
                 patch.object(runner, "_sync"), \
                 patch.object(torch.cuda, "empty_cache"), \
                 patch.object(torch.cuda, "reset_peak_memory_stats"), \
                 patch.object(torch.cuda, "max_memory_reserved", return_value=1), \
                 patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=100)):
                result = runner.measure_capacity(model, optimizer, dataset, device=torch.device("cuda"),
                    rays=1, effective=256, sources={"dl3dv": .2, "arkitscenes": .8}, logger=Mock(),
                    calibration_seconds=seconds, deadline=deadline, should_stop=lambda: stop_after is not None and len(calls) >= stop_after)
        finally:
            self.assertTrue(all(torch.equal(value, original[key]) for key, value in model.state_dict().items()))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            actual_optimizer = optimizer.state_dict()
            self.assertEqual(actual_optimizer["param_groups"], original_optimizer["param_groups"])
            self.assertEqual(set(actual_optimizer["state"]), set(original_optimizer["state"]))
            for key, state in original_optimizer["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(actual_optimizer["state"][key][name], value))
        return result, calls

    def test_capacity_caps_search_at_logical_source_counts_and_restores_state(self):
        (caps, records), calls = self._probe()
        self.assertEqual(calls[:2], [("dl3dv", 1), ("arkitscenes", 1)])
        self.assertEqual(caps, {"dl3dv": 51, "arkitscenes": 205})
        self.assertEqual(max(n for s, n in calls if s == "dl3dv"), 51)
        self.assertEqual(max(n for s, n in calls if s == "arkitscenes"), 205)

    def test_capacity_budget_keeps_only_measured_safe_capacity(self):
        (caps, records), calls = self._probe(seconds=3)
        self.assertEqual(calls, [("dl3dv", 1), ("arkitscenes", 1), ("dl3dv", 2)])
        self.assertEqual(caps, {"dl3dv": 2, "arkitscenes": 1})

    def test_capacity_budget_without_all_sources_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "no safe measured microbatch.*arkitscenes"):
            self._probe(seconds=1)

    def test_capacity_stop_before_all_sources_is_clean_interruption(self):
        from mavt4d.object_state.v5.runner import CalibrationStopped
        with self.assertRaises(CalibrationStopped):
            self._probe(stop_after=1)

    def test_capacity_stop_after_baseline_keeps_complete_measured_caps(self):
        (caps, records), calls = self._probe(stop_after=2)
        self.assertEqual(caps, {"dl3dv": 1, "arkitscenes": 1})
        self.assertEqual(len(calls), 2)

    def test_allocation_deadline_bounds_calibration_even_with_larger_budget(self):
        (caps, records), calls = self._probe(seconds=900, deadline=2)
        self.assertEqual(caps, {"dl3dv": 1, "arkitscenes": 1})
        self.assertEqual(len(calls), 2)

    def test_capacity_rejects_nonfinite_loss(self):
        with self.assertRaisesRegex(FloatingPointError, "nonfinite.*capacity"):
            self._probe(nonfinite="loss")

    def test_capacity_rejects_nonfinite_gradient(self):
        with self.assertRaisesRegex(FloatingPointError, "nonfinite.*capacity"):
            self._probe(nonfinite="gradient")

    def test_capacity_restores_prefilled_adam_state(self):
        self._probe(prefilled=True)


if __name__ == "__main__":
    unittest.main()
