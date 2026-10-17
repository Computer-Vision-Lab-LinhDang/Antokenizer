"""Losses count valid evidence, not padding; pose matching is loss-side only."""
import torch

from mavt4d.object_state.objectives import (
    masked_mean, region_balanced_rgb_loss, readout_loss,
    matched_pose_loss, weighted_objective, association_loss, existence_loss,
)


def test_masked_mean_is_per_sample_and_ignores_nan_padding():
    x = torch.tensor([[1., 3., float("nan")], [5., float("nan"), float("nan")]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    loss = masked_mean(x, mask)
    assert loss == 3.5
    loss.backward()
    assert torch.isfinite(x.grad).all()
    empty = masked_mean(x, torch.zeros_like(mask))
    assert empty == 0 and torch.isfinite(empty)


def test_region_balance_does_not_reward_background_dominance():
    pred = torch.zeros(1, 10, 3, requires_grad=True)
    target = torch.zeros_like(pred)
    target[:, :1] = 1
    instance = torch.zeros(1, 10, dtype=torch.int64)
    instance[:, :1] = 1
    loss = region_balanced_rgb_loss(pred, target, instance)
    assert torch.allclose(loss, torch.tensor(.5))
    loss.backward()
    assert pred.grad[:, 0].abs().sum() > 0


def test_readout_terms_are_zero_for_exact_targets_and_empty_depth_safe():
    gt = {"rgb": torch.rand(2, 8, 3), "range": torch.ones(2, 8),
          "depth_valid": torch.ones(2, 8, dtype=torch.bool),
          "instance": torch.zeros(2, 8, dtype=torch.int64),
          "valid": torch.ones(2, 8, dtype=torch.bool)}
    pred = {"rgb": gt["rgb"].clone().requires_grad_(), "range": gt["range"].clone().requires_grad_(),
            "opacity": torch.ones(2, 8, requires_grad=True)}
    losses = readout_loss(pred, gt)
    assert all(v == 0 for v in losses.values())
    gt["depth_valid"].zero_()
    gt["range"].fill_(float("nan"))
    assert torch.isfinite(readout_loss(pred, gt)["depth"])
    total, parts = weighted_objective(losses, {"rgb": 1., "depth": .1, "opacity": .1})
    total.backward()
    assert set(parts) == {"rgb", "depth", "opacity"}


def test_pose_matching_is_permutation_invariant_and_differentiable():
    gt = torch.eye(4).repeat(1, 2, 1, 1)
    gt[0, :, 0, 3] = torch.tensor([1., 3.])
    pred = gt[:, [1, 0]].clone().requires_grad_()
    valid = torch.ones(1, 2, dtype=torch.bool)
    loss, match = matched_pose_loss(pred, gt, valid)
    assert loss == 0 and torch.equal(match, torch.tensor([[1, 0]]))
    shifted = pred + torch.nn.functional.pad(torch.ones(1, 2, 3, 1) * .1, (3, 0, 0, 1))
    loss, _ = matched_pose_loss(shifted, gt, valid)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    empty, match = matched_pose_loss(pred, gt, torch.zeros_like(valid))
    assert empty == 0 and (match == -1).all()


def test_association_penalizes_all_background_and_masks_unmatched_objects():
    # GT object1 -> slot1, object2 absent/unmatched; final column is background.
    labels = torch.tensor([[1, 1, 0, -1, 2]])
    matching = torch.tensor([[1, -1]])
    logits = torch.tensor([[[-6., 6., -6.], [-6., 6., -6.], [-6., -6., 6.],
                            [-6., -6., 6.], [0., 0., 0.]]], requires_grad=True)
    good = association_loss(logits.softmax(-1), labels, matching)
    all_bg = association_loss(torch.tensor([[[.01, .01, .98]]]).expand(1, 5, 3), labels, matching)
    assert good < .001 and all_bg > 2
    good.backward()
    assert logits.grad[0, 4].abs().sum() == 0
    assert torch.isfinite(logits.grad).all()


def test_existence_penalizes_duplicate_slots_without_oracle_feedback():
    matching = torch.tensor([[1, -1]])
    good = torch.tensor([[.01, .99, .01]], requires_grad=True)
    duplicate = torch.tensor([[.99, .99, .99]], requires_grad=True)
    assert existence_loss(good, matching) < existence_loss(duplicate, matching)
    existence_loss(duplicate, matching).backward()
    assert duplicate.grad[0, 0] > 0 and duplicate.grad[0, 1] < 0


def test_real_four_block_forward_backward_uses_dataset_contract():
    from mavt4d.object_state.config import ObjectStateConfig
    from mavt4d.object_state.model import ObjectStateModel
    from mavt4d.object_state.data import (
        FactorialSceneConfig, FactorialSceneDataset, flatten_source_batch,
        observation_step, sample_ray_targets, patch_instance_targets,
    )
    torch.manual_seed(7)
    cfg = ObjectStateConfig(slots=4, width=16, evidence_width=24, spatial_heads=3,
                            spatial_blocks=1, content_tokens=2, dynamic_tokens=2,
                            background_tokens=2, patch_size=8, samples_per_ray=4,
                            relay_tokens=2, candidates=8)
    ds = FactorialSceneDataset(1, FactorialSceneConfig(frames=6, prefix_frames=4,
                                height=16, width=24, max_objects=2))
    batch = flatten_source_batch(torch.utils.data.default_collate([ds[0]]))
    model = ObjectStateModel(cfg)
    state = model.initial_state(8)
    for frame in (1, 3):
        state, details = model(state, observation_step(batch, frame),
                               return_diagnostics=True, candidate_budget=0)
    targets = sample_ray_targets(batch, 3, 16)
    predicted = model.read(state, targets["query"])
    terms = readout_loss(predicted, targets)
    pose, matching = matched_pose_loss(predicted["controls"], targets["object_poses"],
                                       targets["object_valid"], state.allocated)
    labels, valid = patch_instance_targets(batch, 3, patch_size=8)
    total = sum(terms.values()) + pose + association_loss(details["ownership"], labels, matching, valid)
    total = total + existence_loss(state.p_exist, matching)
    future = sample_ray_targets(batch, 5, 16)
    prior = model.predict(state, future["query"].time - state.t)
    total = total + sum(readout_loss(model.read(prior, future["query"]), future).values())
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(p.grad.abs().sum()) for p in model.dynamics.parameters() if p.grad is not None) > 0
