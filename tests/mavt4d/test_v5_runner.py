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


if __name__ == "__main__": unittest.main()
