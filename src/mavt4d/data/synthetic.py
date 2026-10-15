"""Synthetic driving-like scenes with exact ground truth.

A ground plane z = 0 plus yawed boxes, ray-cast in torch for every camera pixel and LiDAR beam. The world
frame is the ego frame at frame 0 (x forward, y left, z up); the ego drives straight along +x. Colours are
a Lambertian texture of the object-frame point, so a physical point has the same colour in every view.
Moving boxes are placed in front of the ego, inside the central camera's view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mavt4d.data.sample import Boxes3D, SceneSample
from mavt4d.geometry.boxes import half_extents, move_with_boxes, rot_z, world_to_box
from mavt4d.geometry.camera import project

CAM_HEIGHT, CAM_RING, CAM_PITCH_DEG = 1.5, 0.5, 15.0
LIDAR_HEIGHT, LIDAR_MOUNT_YAW = 1.8, math.pi / 2
MAX_RANGE = 80.0
SKY = (-0.2, 0.2, 0.8)
_TEX_A = torch.tensor([[1.0, 0.3, 0.7], [0.5, 1.0, 0.2], [0.2, 0.6, 1.0]])
_TEX_B = torch.tensor([[0.4, 1.0, 0.3], [1.0, 0.2, 0.8], [0.6, 0.5, 0.9]])


@dataclass
class SyntheticSceneConfig:
    V: int = 3
    T: int = 4
    H: int = 64
    W: int = 112
    patch: int = 16
    n_boxes: int = 3
    n_moving: int = 1
    ego_speed: float = 5.0
    dt: float = 1 / 12
    cam_yaw_step_deg: float = 50.0
    fov_deg: float = 70.0
    lidar_beams: int = 16
    lidar_az: int = 256
    lidar_elev_deg: tuple = (-25.0, 5.0)


@dataclass
class SyntheticGT:
    world_xyz: Tensor  # (V, T, H, W, 3)
    depth: Tensor  # (V, T, H, W) camera-frame z
    instance: Tensor  # (V, T, H, W) int64; 0 = ground
    valid: Tensor  # (V, T, H, W) bool; False = sky


def texture(p_obj: Tensor, instance: Tensor) -> Tensor:
    """Colour (N, 3) in [-0.9, 0.9] of object-frame points (world frame for the ground, instance 0)."""
    k = instance.to(p_obj.dtype)[:, None]
    a, b = p_obj @ _TEX_A.to(p_obj), p_obj @ _TEX_B.to(p_obj)
    return 0.5 * torch.sin(1.3 * a + 0.9 * k) + 0.4 * torch.sin(2.9 * b + 2.3 * k)


def _raycast(o: Tensor, d: Tensor, boxes: Boxes3D) -> tuple[Tensor, Tensor]:
    """Nearest hit along rays o + t d (N, 3): t (inf = miss) and instance (0 = ground, -1 = miss)."""
    t = torch.where(d[:, 2] < 0, -o[:, 2] / d[:, 2], torch.full_like(o[:, 2], math.inf))
    inst = torch.zeros(o.shape[0], dtype=torch.int64)
    for i in range(boxes.center.shape[0]):
        R = rot_z(boxes.yaw[i])
        ol, dl = world_to_box(o, boxes.center[i], boxes.yaw[i]), d @ R
        h = half_extents(boxes.size[i])
        t1, t2 = (-h - ol) / dl, (h - ol) / dl
        near, far = torch.minimum(t1, t2).amax(-1), torch.maximum(t1, t2).amin(-1)
        hit = (near <= far) & (near > 1e-6) & (near < t)
        t, inst = torch.where(hit, near, t), torch.where(hit, boxes.instance_id[i], inst)
    miss = ~(t * d.norm(dim=-1) <= MAX_RANGE)
    return t.masked_fill(miss, math.inf), inst.masked_fill(miss, -1)


def _make_boxes(cfg: SyntheticSceneConfig, g: torch.Generator) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initial centres, sizes (w, l, h), yaws and velocities of non-overlapping boxes on the ground."""
    def u(lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=g, dtype=torch.float64))

    fan = math.radians(cfg.cam_yaw_step_deg * (cfg.V - 1) / 2 + cfg.fov_deg / 2 - 12)
    travel = 6.0 * cfg.T * cfg.dt  # max box displacement over the clip
    centers, sizes, yaws, vels = [], [], [], []
    for _ in range(1000):
        if len(centers) == cfg.n_boxes:
            break
        moving = len(centers) < cfg.n_moving
        r, az = (u(8, 13), u(-0.25, 0.25)) if moving else (u(8, 18), u(-fan, fan))
        w, l, h, yaw = u(1.6, 2.2), u(3.5, 4.8), u(1.4, 2.0), u(-math.pi, math.pi)
        c = torch.tensor([r * math.cos(az), r * math.sin(az), h / 2], dtype=torch.float64)
        radius = math.hypot(w, l) / 2
        if all((c[:2] - c2[:2]).norm() > radius + math.hypot(s2[0], s2[1]) / 2 + 2 * travel + 0.5
               for c2, s2 in zip(centers, sizes)):
            speed = u(3, 6) if moving else 0.0
            centers.append(c)
            sizes.append(torch.tensor([w, l, h], dtype=torch.float64))
            yaws.append(yaw)
            vels.append(torch.tensor([speed * math.cos(yaw), speed * math.sin(yaw), 0.0], dtype=torch.float64))
    if len(centers) < cfg.n_boxes:
        raise ValueError(f"n_boxes: could not place {cfg.n_boxes} non-overlapping boxes")
    if not centers:
        z = torch.zeros(0, 3, dtype=torch.float64)
        return z, z.clone(), torch.zeros(0, dtype=torch.float64), z.clone()
    return torch.stack(centers), torch.stack(sizes), torch.tensor(yaws, dtype=torch.float64), torch.stack(vels)


def _camera_rotations(cfg: SyntheticSceneConfig) -> Tensor:
    """(V, 3, 3) camera-to-world rotations; optical frame x right, y down, z forward."""
    out = []
    pitch = math.radians(CAM_PITCH_DEG)
    for v in range(cfg.V):
        yaw = math.radians((v - (cfg.V - 1) / 2) * cfg.cam_yaw_step_deg)
        fwd = torch.tensor([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)])
        right = torch.tensor([math.sin(yaw), -math.cos(yaw), 0.0])
        out.append(torch.stack([right, torch.linalg.cross(fwd, right), fwd], 1))
    return torch.stack(out).double()


def make_synthetic_scene(cfg: SyntheticSceneConfig, seed: int) -> tuple[SceneSample, SyntheticGT]:
    """Render one scene (tier A: cameras, one LiDAR sweep per frame, per-frame boxes) and its exact GT."""
    g = torch.Generator().manual_seed(seed)
    V, T, H, W = cfg.V, cfg.T, cfg.H, cfg.W
    c0, size, yaw, vel = _make_boxes(cfg, g)
    times = torch.arange(T, dtype=torch.float64) * cfg.dt
    boxes64 = [Boxes3D(center=c0 + vel * float(times[t]), size=size, yaw=yaw,
                       instance_id=torch.arange(1, len(c0) + 1)) for t in range(T)]
    boxes = [Boxes3D(center=b.center.float(), size=b.size.float(), yaw=b.yaw.float(), instance_id=b.instance_id)
             for b in boxes64]
    ego = torch.stack([times * cfg.ego_speed, torch.zeros(T, dtype=torch.float64),
                       torch.zeros(T, dtype=torch.float64)], 1)  # (T, 3)

    f = (W / 2) / math.tan(math.radians(cfg.fov_deg) / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float64)
    R = _camera_rotations(cfg)
    c2w = torch.zeros(V, T, 4, 4, dtype=torch.float64)
    c2w[..., :3, :3] = R[:, None]
    yaws = torch.tensor([(v - (V - 1) / 2) * math.radians(cfg.cam_yaw_step_deg) for v in range(V)],
                        dtype=torch.float64)
    ring = torch.stack([torch.cos(yaws), torch.sin(yaws), torch.zeros(V, dtype=torch.float64)], 1) * CAM_RING
    c2w[..., :3, 3] = ego[None] + ring[:, None] + torch.tensor([0, 0, CAM_HEIGHT], dtype=torch.float64)
    c2w[..., 3, 3] = 1

    jj, ii = torch.meshgrid(torch.arange(W, dtype=torch.float64), torch.arange(H, dtype=torch.float64),
                            indexing="xy")
    pix = torch.stack([jj + 0.5, ii + 0.5, torch.ones_like(jj)], -1).reshape(-1, 3)  # (H*W, 3), z_cam = 1
    rays_cam = pix @ torch.linalg.inv(K).T

    world = torch.zeros(V, T, H * W, 3, dtype=torch.float64)
    depth = torch.zeros(V, T, H * W, dtype=torch.float64)
    inst = torch.zeros(V, T, H * W, dtype=torch.int64)
    for v in range(V):
        for t in range(T):
            o = c2w[v, t, :3, 3].expand(H * W, 3)
            d = rays_cam @ c2w[v, t, :3, :3].T
            hit_t, hit_i = _raycast(o, d, boxes64[t])
            ok = hit_i >= 0
            world[v, t][ok] = o[ok] + hit_t[ok, None] * d[ok]
            depth[v, t][ok] = hit_t[ok]  # rays have unit camera z, so t is depth
            inst[v, t] = hit_i
    valid = inst >= 0
    inst = inst.clamp(min=0)
    world32 = world.float()

    images = torch.tensor(SKY).view(1, 1, 1, 3).repeat(V, T, H * W, 1)
    for v in range(V):
        for t in range(T):
            ok = valid[v, t]
            p_obj = world32[v, t][ok].clone()
            k = inst[v, t][ok]
            for i in range(len(c0)):
                sel = k == boxes[t].instance_id[i]
                p_obj[sel] = world_to_box(p_obj[sel], boxes[t].center[i], boxes[t].yaw[i])
            images[v, t][ok] = texture(p_obj, k)

    elev = torch.deg2rad(torch.linspace(*cfg.lidar_elev_deg, cfg.lidar_beams, dtype=torch.float64))
    az = torch.arange(cfg.lidar_az, dtype=torch.float64) * (2 * math.pi / cfg.lidar_az)
    e, a = torch.meshgrid(elev, az, indexing="ij")
    dirs_l = torch.stack([e.cos() * a.cos(), e.cos() * a.sin(), e.sin()], -1).reshape(-1, 3)
    R_l = rot_z(torch.tensor(LIDAR_MOUNT_YAW, dtype=torch.float64))
    l2w = torch.eye(4, dtype=torch.float64).repeat(T, 1, 1)
    l2w[:, :3, :3] = R_l
    l2w[:, :3, 3] = ego + torch.tensor([0, 0, LIDAR_HEIGHT], dtype=torch.float64)
    lidar = []
    for t in range(T):
        hit_t, hit_i = _raycast(l2w[t, :3, 3].expand_as(dirs_l), dirs_l @ R_l.T, boxes64[t])
        ok = hit_i >= 0
        xyz = hit_t[ok, None] * dirs_l[ok]  # unit LiDAR-frame directions: t is range
        lidar.append(torch.cat([xyz, (0.1 + 0.15 * hit_i[ok, None]).double()], 1).float())

    sample = SceneSample(
        scene_id=f"synthetic-{seed}", source="synthetic", images=images.reshape(V, T, H, W, 3).permute(0, 1, 4, 2, 3)
        .contiguous(), frame_times=times.expand(V, T).clone(), intrinsics=K.float().expand(V, 3, 3).clone(),
        cam_to_world=c2w.float(), lidar_points=lidar, lidar_to_world=l2w.float(), lidar_times=times.clone(),
        lidar_keys=[f"seed{seed}/sweep{t}" for t in range(T)], boxes=boxes)
    gt = SyntheticGT(world_xyz=world32.reshape(V, T, H, W, 3), depth=depth.float().reshape(V, T, H, W),
                     instance=inst.reshape(V, T, H, W), valid=valid.reshape(V, T, H, W))
    return sample, gt


def gt_visible(uv: Tensor, z: Tensor, depth: Tensor, valid: Tensor) -> Tensor:
    """(N,) bool: points at pixels ``uv`` with camera depth ``z`` are inside the image and not behind the GT
    surface. The reference is the farthest of the four pixel centres around ``uv`` (sky = infinitely far),
    so steep depth gradients (grazing ground) do not count as occlusion."""
    H, W = depth.shape
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    far = torch.where(valid, depth, torch.full_like(depth, math.inf))
    u0, v0 = (uv[:, 0] - 0.5).floor().long(), (uv[:, 1] - 0.5).floor().long()
    ref = torch.full_like(z, -math.inf)
    for du in (0, 1):
        for dv in (0, 1):
            ref = torch.maximum(ref, far[(v0 + dv).clamp(0, H - 1), (u0 + du).clamp(0, W - 1)])
    return inside & (z <= 1.02 * ref + 0.05)


def gt_patch_flow(sample: SceneSample, gt: SyntheticGT, cfg: SyntheticSceneConfig) -> Tensor:
    """(V, T-1, Hp, Wp, 3) forward flow of each patch centre in patch units, plus valid.

    The centre's motion is the mean motion of the 2x2 pixels around it; valid needs all four to be scene
    points that are visible at t+1 (``gt_visible``).
    """
    V, T, H, W, p = cfg.V, cfg.T, cfg.H, cfg.W, cfg.patch
    Hp, Wp = H // p, W // p
    rows = (torch.arange(Hp)[:, None] * p + p // 2 + torch.tensor([-1, 0])).flatten()  # (Hp*2,)
    cols = (torch.arange(Wp)[:, None] * p + p // 2 + torch.tensor([-1, 0])).flatten()  # (Wp*2,)
    uv0 = torch.stack(torch.meshgrid(cols.float() + 0.5, rows.float() + 0.5, indexing="xy"), -1).reshape(-1, 2)
    out = torch.zeros(V, T - 1, Hp, Wp, 3)
    for v in range(V):
        for t in range(T - 1):
            sub = (slice(None), cols)
            X = gt.world_xyz[v, t][rows][sub].reshape(-1, 3)
            k = gt.instance[v, t][rows][sub].flatten()
            ok = gt.valid[v, t][rows][sub].flatten()
            X1 = move_with_boxes(X, torch.where(k > 0, k, -1), sample.boxes[t], sample.boxes[t + 1])
            uv, z = project(X1, sample.intrinsics[v], sample.cam_to_world[v, t + 1])
            seen = gt_visible(uv, z, gt.depth[v, t + 1], gt.valid[v, t + 1])
            disp = ((uv - uv0) / p).reshape(Hp, 2, Wp, 2, 2).mean((1, 3))
            good = (ok & seen).reshape(Hp, 2, Wp, 2).all(3).all(1)
            out[v, t] = torch.cat([disp, good[..., None].float()], -1)
    return out


class SyntheticSceneDataset(Dataset):
    """``n_scenes`` scenes rendered on demand from seeds ``base_seed + i``."""

    def __init__(self, n_scenes: int, cfg: SyntheticSceneConfig, base_seed: int = 0) -> None:
        self.n_scenes, self.cfg, self.base_seed = n_scenes, cfg, base_seed

    def __len__(self) -> int:
        return self.n_scenes

    def __getitem__(self, i: int) -> SceneSample:
        return make_synthetic_scene(self.cfg, self.base_seed + i)[0]

    def ground_truth(self, i: int) -> SyntheticGT:
        return make_synthetic_scene(self.cfg, self.base_seed + i)[1]
