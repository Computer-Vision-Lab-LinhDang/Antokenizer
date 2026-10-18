import torch

from mavt4d.object_state.v5.evidence import (
    atlas_lookup, atlas_update, classify_visibility, transport_rigid_points, visibility_from_depth,
)


def test_spatial_support_separates_regions_and_preserves_duplicate_bit_writes():
    bits = torch.zeros(1, 1, 64, dtype=torch.uint8)
    points = torch.tensor([[[[-3.5, -3.5, -3.5], [-2.5, -3.5, -3.5], [-3.5, -3.5, -3.5]]]])
    result = atlas_update(bits, points, torch.ones(1, 1, 3, dtype=torch.bool))
    assert bits.sum() == 0
    assert torch.equal(atlas_lookup(result, points), torch.ones(1, 1, 3))
    query = torch.tensor([[[[3.5, 3.5, 3.5], [float('nan'), 0, 0], [4.1, 0, 0]]]])
    assert atlas_lookup(result, query).sum() == 0
    assert torch.equal(result, atlas_update(result, points, torch.ones(1, 1, 3, dtype=torch.bool)))
    assert result.numel() * result.element_size() == 64


def test_missing_history_never_certifies_unknown():
    visible = torch.tensor([[[False, False], [True, False], [False, True], [False, False]]])
    checked = torch.tensor([[[False, True], [True, False], [False, True], [True, True]]])
    assert classify_visibility(visible, checked, current_index=1).tolist() == [[-1, -1, 0, 2]]
    assert classify_visibility(visible, checked, current_index=None).tolist() == [[-1, 1, 1, 2]]


def test_rigid_surface_history_tracks_material_point_not_world_position():
    current = torch.eye(4).reshape(1, 4, 4)
    current[:, 0, 3] = 5
    past = torch.eye(4).reshape(1, 1, 4, 4)
    points = torch.tensor([[[6., 2., 3.]]])
    transported = transport_rigid_points(points, current, past)
    assert transported.shape == (1, 1, 1, 3)
    assert torch.allclose(transported[0, 0, 0], torch.tensor([1., 2., 3.]))


def test_untrusted_visibility_is_rejected():
    visible = torch.ones(1, 1, 1, dtype=torch.bool)
    checked = torch.zeros_like(visible)
    try:
        classify_visibility(visible, checked, current_index=0)
    except ValueError:
        return
    raise AssertionError('untrusted positive visibility must not become a target')


def test_visibility_distinguishes_masked_outside_missing_and_occluded():
    points = torch.tensor([[[[0., 0., 2.], [20., 0., 2.], [2., 0., 2.], [0., 2., 4.]]]])
    pose = torch.eye(4).reshape(1, 1, 4, 4)
    intrinsic = torch.eye(3).reshape(1, 1, 3, 3)
    depth = torch.full((1, 1, 2, 2), 2.)
    dv = torch.ones_like(depth, dtype=torch.bool)
    dv[..., 0, 1] = False
    validity = torch.ones_like(dv)
    trusted = torch.ones(1, 1, 4, dtype=torch.bool)
    visible, checked = visibility_from_depth(points, pose, intrinsic, depth, dv, validity, trusted)
    assert visible.tolist() == [[[True, False, False, False]]]
    assert checked.tolist() == [[[True, True, False, True]]]
    validity[..., 0, 0] = False
    visible, checked = visibility_from_depth(points, pose, intrinsic, depth, dv, validity, trusted)
    assert not visible.any()
    assert checked[0, 0, 0]
