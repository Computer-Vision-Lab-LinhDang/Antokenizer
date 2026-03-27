# MAVT Dataloader & Training Stages
## Chi tiết thiết kế hệ thống loading, packing, và progressive training

---

## Mục lục

1. Tổng quan kiến trúc Dataloader
2. Per-Modality Dataset Design
3. Module 1+2: Convert to Unified 4D Tokens
4. NaViT-style Token Packing
5. Mask Generation (Attention + Graph + Mamba)
6. Progressive Training Stages
7. Optimizer & Scheduler
8. Pseudocode hoàn chỉnh
9. Compute Budget Estimation

---

## 1. Tổng quan kiến trúc Dataloader

### 1.1 Pipeline tổng thể

```
RAW DATA (per modality, stored riêng)
  │
  │  ImageDataset ──→ (3, H, W)
  │  VideoDataset ──→ (3, T, H, W)
  │  Object3DDataset → (3_planes, C, S, S)
  │
  ▼
CONVERT TO 4D (Module 1 + Module 2, chạy trong Dataset.__getitem__)
  │
  │  Mỗi sample → {
  │      tokens:    (N_i, 1152)     # SigLIP2 features + freq embedding
  │      positions: (N_i, 4)        # (t, x, y, z) coordinates
  │      freq_raw:  (N_i, 15)       # raw frequency profile
  │      modality:  str              # "image"/"video"/"3d"
  │      caption:   str              # text description
  │      N_i:       int              # token count
  │  }
  │
  │  SAU BƯỚC NÀY: tất cả modalities CÙNG FORMAT
  │
  ▼
NAVIT PACKING (trong custom collate_fn)
  │
  │  Nhiều 4D samples → pack vào sequence dài L
  │  Generate masks: attention, graph, mamba
  │
  ▼
BATCH OUTPUT → GPU
  │
  │  tokens:      (B, L, 1152)    # packed token features
  │  positions:   (B, L, 4)       # packed 4D coordinates
  │  freq_raw:    (B, L, 15)      # packed frequency profiles
  │  sample_ids:  (B, L)          # sample assignment (-1 = pad)
  │  attn_mask:   (B, L, L)       # block-diagonal attention mask
  │  graph_mask:  (B, L, L)       # per-sample graph isolation
  │  sample_meta: list[dict]      # modality, caption, boundaries
  │
  ▼
MODEL FORWARD (GPU)
```

### 1.2 Nguyên tắc thiết kế

```
1. CONVERT EARLY:    Chuyển về 4D tokens TRƯỚC KHI batching
                     → model không biết modality
                     
2. PACK DENSE:       NaViT-style packing, nhiều samples per sequence
                     → GPU utilization cao, ít padding waste
                     
3. ISOLATE SAMPLES:  Masks đảm bảo tokens khác sample không interact
                     → Attention: block-diagonal mask
                     → Graph: no cross-sample edges
                     → Mamba: state reset giữa samples
                     
4. STAGE PROGRESSIVE: Thêm modality theo stages
                      → Training stability, feature quality
                      
5. WEIGHT CONTROLLED: Sampling weights kiểm soát modality ratio
                      → Follow AToken's proven ratios
```

---

## 2. Per-Modality Dataset Design

### 2.1 ImageDataset

```python
class ImageDataset(Dataset):
    """
    Sources:
        - DFN-2B (DataComp Filtering Network): 2B image-text pairs
        - Open Images v7: 9M images, rich annotations
        - LAION-aesthetic: filtered subset, aesthetic score > 5.0
    
    Storage: WebDataset format (.tar shards)
        - Mỗi shard: ~1000 images, ~500MB
        - Streaming-compatible: không cần download toàn bộ
        - Shuffle: inter-shard + intra-shard shuffling
    
    Pre-filtering (offline, 1 lần):
        - NSFW filter (CLIP-based)
        - Watermark detection
        - Aesthetic score > 5.0
        - Min resolution: 64×64
        - Max aspect ratio: 3:1
        - Dedup: perceptual hash + near-duplicate removal
    
    Output per sample:
        image:   (3, H, W)    float32, range [-1, 1]
        caption: str           text description
        height:  int           original height
        width:   int           original width
    """
    
    def __init__(self, config):
        self.data_paths = config.image_data_paths
        self.resolution_range = config.image_resolution_range  # (64, 512) stage 1
        
        # Resolution buckets cho multi-scale training
        # Chọn resolution gần nhất để minimize resize distortion
        self.buckets = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
        
        # Augmentations (nhẹ — tokenizer cần learn true distribution)
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            # KHÔNG dùng: RandomRotation, RandomErasing, Cutout
            # (tokenizer cần reconstruct faithful, không augment mạnh)
        ])
    
    def __getitem__(self, idx):
        # Load image
        img = self.decode_image(idx)  # PIL Image
        
        # Select target resolution (random within stage range)
        min_res, max_res = self.resolution_range
        available_buckets = [b for b in self.buckets if min_res <= b <= max_res]
        target_res = random.choice(available_buckets)
        
        # Resize + center crop (giữ aspect ratio gần nhất)
        img = self.resize_and_crop(img, target_res)
        
        # Augment
        img = self.augment(img)
        
        # Normalize to [-1, 1]
        img = transforms.ToTensor()(img)       # [0, 1]
        img = img * 2 - 1                       # [-1, 1]
        
        caption = self.get_caption(idx)
        
        return {
            "image": img,           # (3, H, W)
            "caption": caption,
            "modality": "image",
            "height": target_res,
            "width": target_res,
        }
    
    def resize_and_crop(self, img, target_res):
        """
        Resize sao cho cạnh ngắn = target_res, rồi center crop.
        Giữ content tốt hơn random crop cho tokenizer training.
        
        Ví dụ: ảnh 800×600, target=256
          → Resize 800×600 → 341×256 (cạnh ngắn = 256)
          → Center crop 256×256
        """
        w, h = img.size
        scale = target_res / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # Center crop
        left = (new_w - target_res) // 2
        top = (new_h - target_res) // 2
        img = img.crop((left, top, left + target_res, top + target_res))
        return img
```

### 2.2 VideoDataset

```python
class VideoDataset(Dataset):
    """
    Sources:
        - WebVid-10M: 10M video-text pairs, ~52s average duration
        - Panda-70M: 70M videos, high-quality captions
        - InternVid-10M: curated, diverse categories
    
    Storage:
        - Individual .mp4 clips (pre-cut scenes)
        - Metadata JSON: fps, duration, resolution, caption, scene_boundaries
        - Index file: fast random access
    
    Pre-processing (offline, expensive):
        - Scene detection + split (PySceneDetect)
        - Filter: static videos (optical flow magnitude < threshold)
        - Filter: text overlays, watermarks (OCR detection)
        - Filter: very short (< 1s) hoặc very long (> 30s)
        - Extract keyframes + store metadata
    
    Temporal sampling strategy:
        - UNIFORM: chọn T frames equally spaced
        - T phải cùng cho tất cả videos trong micro-batch
        - Nhưng NaViT packing cho phép T khác nhau giữa samples!
    
    Output per sample:
        video:   (3, T, H, W)  float32, range [-1, 1]
        caption: str
        n_frames: int
    """
    
    def __init__(self, config):
        self.frame_counts = [4, 8, 16, 32]  # allowed frame counts
        self.resolution_range = config.video_resolution_range  # (64, 512) stage 2
        self.temporal_patch = config.temporal_patch_size  # τ = 2
        
    def __getitem__(self, idx):
        video_path, metadata = self.index[idx]
        total_frames = metadata["total_frames"]
        fps = metadata["fps"]
        
        # Select frame count (random, nhưng phải chia hết cho temporal_patch)
        max_possible_T = min(total_frames, max(self.frame_counts))
        available_T = [t for t in self.frame_counts if t <= max_possible_T]
        T = random.choice(available_T) if available_T else self.frame_counts[0]
        
        # T phải chia hết cho temporal_patch
        T = (T // self.temporal_patch) * self.temporal_patch
        T = max(T, self.temporal_patch)  # minimum = temporal_patch size
        
        # Uniform temporal sampling
        if total_frames >= T:
            frame_indices = np.linspace(0, total_frames - 1, T, dtype=int)
        else:
            # Video ngắn hơn T frames: repeat last frame
            frame_indices = list(range(total_frames))
            frame_indices += [total_frames - 1] * (T - total_frames)
        
        # Decode frames
        frames = self.decode_frames(video_path, frame_indices)  # (T, H_orig, W_orig, 3)
        
        # Select resolution
        min_res, max_res = self.resolution_range
        available_buckets = [b for b in [64, 128, 256, 384, 512] if min_res <= b <= max_res]
        target_res = random.choice(available_buckets)
        
        # Resize + crop (SAME crop cho tất cả frames — spatial consistency)
        crop_params = self.get_random_crop_params(frames[0], target_res)
        frames = torch.stack([
            self.crop_and_resize(f, crop_params, target_res) for f in frames
        ])  # (T, 3, H, W)
        
        # Reorder to (3, T, H, W) và normalize
        frames = frames.permute(1, 0, 2, 3)  # (3, T, H, W)
        frames = frames * 2 - 1  # normalize [-1, 1]
        
        # Light augmentation (temporal-consistent)
        frames = self.augment_video(frames)  # flip, color jitter (same per frame)
        
        return {
            "video": frames,          # (3, T, H, W)
            "caption": metadata.get("caption", ""),
            "modality": "video",
            "n_frames": T,
            "height": target_res,
            "width": target_res,
        }
    
    def augment_video(self, frames):
        """
        Augmentation phải temporal-consistent:
        flip cùng direction, color jitter cùng params cho tất cả frames.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            frames = frames.flip(-1)  # flip width dimension
        
        # Color jitter (same params cho toàn video)
        brightness = 1.0 + random.uniform(-0.05, 0.05)
        frames = frames * brightness
        frames = frames.clamp(-1, 1)
        
        return frames
```

### 2.3 Object3DDataset

```python
class Object3DDataset(Dataset):
    """
    Sources:
        - Objaverse: ~800K 3D objects (CC-BY license)
        - Objaverse-XL: 10M+ objects (extended)
        - Cap3D: captions cho Objaverse objects
    
    Storage:
        - Pre-rendered multi-view images: {obj_id}/view_{i}.png (i=0..N_views-1)
        - Pre-computed triplanes (optional cache): {obj_id}/triplane.pt
        - Camera poses: {obj_id}/cameras.json
        - Metadata: {obj_id}/meta.json (category, caption, bbox)
    
    Pre-processing (OFFLINE, very expensive — run once):
        1. Load mesh/point cloud
        2. Normalize to unit cube [-0.5, 0.5]³
        3. Sample N_views camera positions on sphere:
           - Radius: 1.5 (unit cube fits in view)
           - Elevation: uniform [-30°, 30°] (mostly horizontal views)
           - Azimuth: uniform [0°, 360°]
           - Resolution: 256×256 per view
        4. Render RGB + depth + normal maps per view
        5. Save rendered images
        6. (Optional) Pre-compute triplane via trained projector
    
    Runtime:
        - Load pre-rendered views
        - Convert to triplane (if not cached) hoặc load cached
        - Triplane = 3 orthogonal feature planes (XY, XZ, YZ)
    
    Oversampling strategy:
        800K objects ≪ 100M images → oversample
        AToken: 3D tasks = 22.2% of training → ~4.5× oversample
        Augmentation: random camera rotation, color jitter per view
    
    Output:
        triplane:    (3, 3, S, S)     # 3 planes × RGB(3) × S×S
        views:       (N_views, 3, H, W)  # for cross-attention (Module 8)
        caption:     str
        cameras:     (N_views, 4, 4)  # camera pose matrices
    """
    
    def __init__(self, config):
        self.n_views = config.n_views          # 8
        self.triplane_res = config.triplane_res  # 32
        self.use_cache = config.use_triplane_cache
        self.render_dir = config.render_dir
        
    def __getitem__(self, idx):
        obj_id = self.index[idx]
        
        # ── Load pre-rendered views ──
        views = []
        cameras = []
        for v in range(self.n_views):
            view = self.load_image(f"{self.render_dir}/{obj_id}/view_{v}.png")
            cam = self.load_camera(f"{self.render_dir}/{obj_id}/cameras.json", v)
            views.append(view)
            cameras.append(cam)
        
        views = torch.stack(views)       # (N_views, 3, 256, 256)
        cameras = torch.stack(cameras)   # (N_views, 4, 4)
        
        # ── 3D Augmentation ──
        # Random rotation around vertical axis (Y)
        rotation_angle = random.uniform(0, 2 * math.pi)
        views, cameras = self.rotate_views(views, cameras, rotation_angle)
        
        # Color jitter (same per view for consistency)
        views = self.color_jitter_consistent(views)
        
        # ── Create triplane representation ──
        if self.use_cache:
            triplane = self.load_triplane_cache(obj_id)  # (3, C, S, S)
        else:
            # Triplane từ rendered views:
            # Simple approach: project views onto 3 orthogonal planes
            # XY plane = top-down view (chọn view gần nhất với top-down)
            # XZ plane = front view
            # YZ plane = side view
            triplane = self.views_to_simple_triplane(views, cameras)
        
        # Normalize
        triplane = triplane * 2 - 1  # [-1, 1]
        views = views * 2 - 1
        
        return {
            "triplane": triplane,      # (3, 3, S, S)
            "views": views,            # (N_views, 3, 256, 256)
            "cameras": cameras,        # (N_views, 4, 4)
            "caption": self.get_caption(obj_id),
            "modality": "3d",
        }
    
    def views_to_simple_triplane(self, views, cameras):
        """
        Simple triplane construction from rendered views.
        Chọn 3 views gần nhất với 3 canonical directions.
        
        Full cross-attention triplane projection (Module 8) sẽ refine
        trong model forward. Đây chỉ là initialization.
        """
        S = self.triplane_res
        
        # Find views closest to canonical directions
        canonical_dirs = torch.tensor([
            [0, 1, 0],   # top-down (XY plane)
            [0, 0, 1],   # front (XZ plane)
            [1, 0, 0],   # side (YZ plane)
        ], dtype=torch.float)
        
        # Camera forward vectors
        cam_forwards = cameras[:, :3, 2]  # (N_views, 3)
        
        triplane = []
        for canonical in canonical_dirs:
            # Find closest view
            similarities = F.cosine_similarity(
                cam_forwards, canonical.unsqueeze(0), dim=-1
            )
            best_view_idx = similarities.argmax()
            best_view = views[best_view_idx]  # (3, 256, 256)
            
            # Resize to triplane resolution
            plane = F.interpolate(
                best_view.unsqueeze(0), size=(S, S), mode='bilinear'
            ).squeeze(0)
            triplane.append(plane)
        
        return torch.stack(triplane)  # (3, 3, S, S)
```

---

## 3. Module 1+2: Convert to Unified 4D Tokens

### 3.1 Conversion trong Dataset (trước batching)

```python
class Unified4DConverter:
    """
    Convert bất kỳ modality nào → unified 4D token set.
    Chạy TRONG dataset __getitem__ (CPU side).
    
    Output luôn cùng format:
        tokens:    (N, 1152)  — SigLIP2 features + freq embedding
        positions: (N, 4)     — (t, x, y, z) coordinates
        freq_raw:  (N, 15)    — [spatial(7), temporal(4), depth(4)]
        n_tokens:  int        — number of tokens
    """
    
    def __init__(self, config):
        self.patch_size = config.patch_size  # 16
        self.temporal_patch = config.temporal_patch  # 2
        
        # SigLIP2 (frozen, shared across all conversions)
        self.siglip2 = load_siglip2_so400m()
        self.siglip2.eval().requires_grad_(False)
        
        # STF transform (lightweight, can run on CPU)
        self.stf = SpaceTimeFrequencyTransform(config)
        
        # Modality embeddings
        self.mod_embeds = {
            "image": torch.zeros(1152),   # or learned, loaded from checkpoint
            "video": torch.zeros(1152),
            "3d":    torch.zeros(1152),
        }
    
    @torch.no_grad()
    def convert(self, sample: dict) -> dict:
        """
        Main conversion entry point.
        
        Input: raw sample dict from any Dataset
        Output: unified 4D token dict
        """
        modality = sample["modality"]
        
        if modality == "image":
            return self._convert_image(sample)
        elif modality == "video":
            return self._convert_video(sample)
        elif modality == "3d":
            return self._convert_3d(sample)
    
    def _convert_image(self, sample):
        """
        Image (3, H, W) → N tokens in 4D space
        
        Ví dụ: image 256×256, patch_size=16
          N = (256/16)² = 256 tokens
          Mỗi token: pos = (t=0, x=col, y=row, z=0)
        """
        img = sample["image"].unsqueeze(0)  # (1, 3, H, W)
        H, W = img.shape[2], img.shape[3]
        p = self.patch_size
        n_h, n_w = H // p, W // p
        N = n_h * n_w
        
        # ── SigLIP2 patch embedding ──
        features = self.siglip2.patch_embed(img).squeeze(0)  # (N, 1152)
        
        # ── 4D positions ──
        gy, gx = torch.meshgrid(
            torch.arange(n_h, dtype=torch.float),
            torch.arange(n_w, dtype=torch.float),
            indexing='ij'
        )
        positions = torch.zeros(N, 4)
        positions[:, 0] = 0.0                  # t = 0
        positions[:, 1] = gx.flatten()         # x
        positions[:, 2] = gy.flatten()         # y
        positions[:, 3] = 0.0                  # z = 0
        
        # ── Raw patches cho DWT ──
        raw_patches = img.squeeze(0).unfold(1, p, p).unfold(2, p, p)
        raw_patches = raw_patches.permute(1, 2, 0, 3, 4).reshape(N, 3, p, p)
        
        # ── STF: spatial frequency only (image) ──
        freq_embed, freq_raw = self.stf(
            raw_patches.unsqueeze(0),
            temporal_signal=None,
            depth_signal=None
        )
        freq_embed = freq_embed.squeeze(0)  # (N, 128)
        freq_raw = freq_raw.squeeze(0)      # (N, 15) [spatial(7), zeros(4), zeros(4)]
        
        # ── Combine features ──
        # Project freq_embed to same dim and add
        tokens = features  # (N, 1152)
        # freq_embed will be added later in model (after projection)
        
        return {
            "tokens": tokens,          # (N, 1152)
            "positions": positions,    # (N, 4)
            "freq_raw": freq_raw,      # (N, 15)
            "freq_embed": freq_embed,  # (N, 128)
            "n_tokens": N,
            "modality": "image",
            "caption": sample.get("caption", ""),
            "resolution": (H, W),
        }
    
    def _convert_video(self, sample):
        """
        Video (3, T, H, W) → N tokens in 4D space
        
        Ví dụ: video 16 frames × 256×256, temporal_patch=2, spatial_patch=16
          n_t = 16/2 = 8 temporal chunks
          n_h × n_w = (256/16)² = 256 spatial patches per chunk
          N = 8 × 256 = 2048 tokens
          
        Mỗi token: pos = (t=chunk_idx, x=col, y=row, z=0)
        """
        video = sample["video"]  # (3, T, H, W)
        C, T, H, W = video.shape
        p = self.patch_size
        τ = self.temporal_patch
        n_t = T // τ
        n_h, n_w = H // p, W // p
        N_spatial = n_h * n_w
        N_total = n_t * N_spatial
        
        all_features = []
        all_positions = []
        all_raw_patches = []
        per_frame_features = []  # cho temporal STFT
        
        for ti in range(n_t):
            # Average τ frames cho SigLIP2 (image encoder)
            chunk = video[:, ti*τ:(ti+1)*τ].mean(dim=1)  # (3, H, W)
            feat = self.siglip2.patch_embed(chunk.unsqueeze(0)).squeeze(0)  # (N_spatial, 1152)
            all_features.append(feat)
            
            # Positions
            gy, gx = torch.meshgrid(
                torch.arange(n_h, dtype=torch.float),
                torch.arange(n_w, dtype=torch.float),
                indexing='ij'
            )
            pos = torch.zeros(N_spatial, 4)
            pos[:, 0] = float(ti)          # t = temporal chunk index
            pos[:, 1] = gx.flatten()       # x
            pos[:, 2] = gy.flatten()       # y
            pos[:, 3] = 0.0               # z = 0
            all_positions.append(pos)
            
            # Raw patches (center frame of chunk)
            center_frame = video[:, ti*τ + τ//2]  # (3, H, W)
            rp = center_frame.unfold(1, p, p).unfold(2, p, p)
            rp = rp.permute(1, 2, 0, 3, 4).reshape(N_spatial, 3, p, p)
            all_raw_patches.append(rp)
            
            # Per-frame features (full temporal resolution for STFT)
            for fi in range(τ):
                frame = video[:, ti*τ + fi]
                ff = self.siglip2.patch_embed(frame.unsqueeze(0)).squeeze(0)
                per_frame_features.append(ff)
        
        tokens = torch.cat(all_features, dim=0)       # (N_total, 1152)
        positions = torch.cat(all_positions, dim=0)    # (N_total, 4)
        raw_patches = torch.cat(all_raw_patches, dim=0)  # (N_total, 3, p, p)
        
        # Temporal signal cho STFT: (N_spatial, T, D)
        # Reshape per_frame_features to align spatial positions across time
        temporal_signal = torch.stack(per_frame_features, dim=1)  # (N_spatial, T, 1152)
        # Tile for all temporal chunks (each chunk gets signal from all frames)
        temporal_signal_expanded = temporal_signal.unsqueeze(0)  # (1, N_spatial, T, 1152)
        
        # ── STF: spatial + temporal frequency ──
        freq_embed, freq_raw = self.stf(
            raw_patches.unsqueeze(0),
            temporal_signal=temporal_signal_expanded,
            depth_signal=None
        )
        
        # Temporal freq: same for all chunks at same spatial position
        # Expand to match total tokens
        freq_embed_spatial = freq_embed.squeeze(0)[:N_spatial]  # (N_spatial, 128)
        freq_embed_full = freq_embed_spatial.repeat(n_t, 1)      # (N_total, 128)
        freq_raw_spatial = freq_raw.squeeze(0)[:N_spatial]
        freq_raw_full = freq_raw_spatial.repeat(n_t, 1)          # (N_total, 15)
        
        return {
            "tokens": tokens,
            "positions": positions,
            "freq_raw": freq_raw_full,
            "freq_embed": freq_embed_full,
            "n_tokens": N_total,
            "modality": "video",
            "caption": sample.get("caption", ""),
            "resolution": (T, H, W),
        }
    
    def _convert_3d(self, sample):
        """
        3D Triplane (3, 3, S, S) → N tokens in 4D space
        
        3 planes mapped to 4D:
          XY plane (top-down):  pos = (0, x, y, 0)    ← giống image!
          XZ plane (front):     pos = (0, x, 0, z)
          YZ plane (side):      pos = (0, 0, y, z)
        
        Ví dụ: triplane S=32, patch_size=16
          N_per_plane = (32/16)² = 4 tokens
          N_total = 3 × 4 = 12 tokens
          
        Nhưng nếu S=64:
          N_per_plane = (64/16)² = 16
          N_total = 3 × 16 = 48 tokens
          
        Thực tế: S=32 với patch_size=4 (nhỏ hơn cho 3D detail):
          N_per_plane = (32/4)² = 64
          N_total = 3 × 64 = 192 tokens
        """
        triplane = sample["triplane"]  # (3, 3, S, S)
        n_planes, C, S, _ = triplane.shape
        
        # 3D patch size nhỏ hơn (cần detail cho 3D)
        p3d = min(self.patch_size, S // 4)  # e.g., 4 for S=32
        n_s = S // p3d
        N_per_plane = n_s * n_s
        N_total = 3 * N_per_plane
        
        all_features = []
        all_positions = []
        all_raw_patches = []
        
        # Plane configurations: (plane_idx, position_mapping)
        plane_configs = [
            # XY plane: x=col, y=row, z=0
            {"x_dim": 1, "y_dim": 2, "fixed_dim": 3, "fixed_val": 0.0},
            # XZ plane: x=col, y=0, z=row
            {"x_dim": 1, "y_dim": 3, "fixed_dim": 2, "fixed_val": 0.0},
            # YZ plane: x=0, y=col, z=row
            {"x_dim": 2, "y_dim": 3, "fixed_dim": 1, "fixed_val": 0.0},
        ]
        
        for pi, pconfig in enumerate(plane_configs):
            plane = triplane[pi]  # (3, S, S) — RGB image of this plane
            
            # SigLIP2 embedding (treat each plane as image)
            # Resize to SigLIP2's expected input if needed
            if S < 224:
                plane_resized = F.interpolate(
                    plane.unsqueeze(0), size=(224, 224), mode='bilinear'
                )
                feat = self.siglip2.patch_embed(plane_resized).squeeze(0)
                # Subsample features to match actual resolution
                n_sig = int(224 / self.patch_size)
                feat = feat.reshape(n_sig, n_sig, -1)
                feat = F.adaptive_avg_pool2d(
                    feat.permute(2, 0, 1).unsqueeze(0),
                    (n_s, n_s)
                ).squeeze(0).permute(1, 2, 0).reshape(N_per_plane, -1)
            else:
                feat = self.siglip2.patch_embed(plane.unsqueeze(0)).squeeze(0)
            
            all_features.append(feat[:N_per_plane])
            
            # 4D positions
            ga, gb = torch.meshgrid(
                torch.arange(n_s, dtype=torch.float),
                torch.arange(n_s, dtype=torch.float),
                indexing='ij'
            )
            pos = torch.zeros(N_per_plane, 4)
            pos[:, 0] = 0.0  # t = 0 always for 3D
            pos[:, pconfig["x_dim"]] = gb.flatten()
            pos[:, pconfig["y_dim"]] = ga.flatten()
            pos[:, pconfig["fixed_dim"]] = pconfig["fixed_val"]
            all_positions.append(pos)
            
            # Raw patches
            rp = plane.unfold(1, p3d, p3d).unfold(2, p3d, p3d)
            rp = rp.permute(1, 2, 0, 3, 4).reshape(N_per_plane, C, p3d, p3d)
            all_raw_patches.append(rp)
        
        tokens = torch.cat(all_features, dim=0)
        positions = torch.cat(all_positions, dim=0)
        raw_patches = torch.cat(all_raw_patches, dim=0)
        
        # ── Depth signal for STFT ──
        # Extract features along z-axis from XZ and YZ planes
        xz_plane = triplane[1]  # (3, S, S) — dims: (C, x, z)
        yz_plane = triplane[2]  # (3, S, S) — dims: (C, y, z)
        
        # Depth signal: for each x position, features along z
        # Use raw pixel values as proxy
        depth_signal_xz = xz_plane.permute(1, 2, 0).unsqueeze(0)  # (1, S_x, S_z, 3)
        
        # ── STF: spatial + depth frequency ──
        freq_embed, freq_raw = self.stf(
            raw_patches.unsqueeze(0),
            temporal_signal=None,
            depth_signal=depth_signal_xz
        )
        freq_embed = freq_embed.squeeze(0)[:N_total]
        freq_raw = freq_raw.squeeze(0)[:N_total]
        
        return {
            "tokens": tokens,
            "positions": positions,
            "freq_raw": freq_raw,
            "freq_embed": freq_embed,
            "n_tokens": N_total,
            "modality": "3d",
            "caption": sample.get("caption", ""),
            "views": sample.get("views", None),
            "cameras": sample.get("cameras", None),
        }
```

### 3.2 Token count per modality per resolution

```
╔═══════════╦═══════════╦═══════════════════════════════════╗
║ Modality  ║ Config    ║ N_tokens                          ║
╠═══════════╬═══════════╬═══════════════════════════════════╣
║ Image     ║ 64×64     ║ (64/16)² = 16                    ║
║           ║ 128×128   ║ (128/16)² = 64                   ║
║           ║ 256×256   ║ (256/16)² = 256                  ║
║           ║ 512×512   ║ (512/16)² = 1024                 ║
║           ║ 1024×1024 ║ (1024/16)² = 4096                ║
║           ║ 2048×2048 ║ (2048/16)² = 16384               ║
╠═══════════╬═══════════╬═══════════════════════════════════╣
║ Video     ║ 4f×128²   ║ (4/2)×(128/16)² = 2×64 = 128    ║
║           ║ 8f×256²   ║ (8/2)×(256/16)² = 4×256 = 1024  ║
║           ║ 16f×256²  ║ (16/2)×256 = 2048                ║
║           ║ 16f×512²  ║ (16/2)×1024 = 8192               ║
║           ║ 32f×256²  ║ (32/2)×256 = 4096                ║
╠═══════════╬═══════════╬═══════════════════════════════════╣
║ 3D        ║ S=32,p=4  ║ 3×(32/4)² = 3×64 = 192          ║
║           ║ S=64,p=4  ║ 3×(64/4)² = 3×256 = 768         ║
║           ║ S=64,p=8  ║ 3×(64/8)² = 3×64 = 192          ║
╚═══════════╩═══════════╩═══════════════════════════════════╝

Observation:
  - Image 256² = Video 4f×128² = 256 tokens
  - Image 512² = Video 8f×256² = 1024 tokens
  → Khác modality, CÙNG token count → pack tự nhiên!
```

---

## 4. NaViT-style Token Packing

### 4.1 Packing Algorithm

```python
class NaViTTokenPacker:
    """
    Pack nhiều 4D token sets vào 1 sequence dài L.
    
    Config:
        max_seq_len:    4096  (L — maximum tokens per packed sequence)
        max_samples:    32    (tối đa samples per pack, prevent too many tiny samples)
        packing_efficiency_target: 0.85  (>85% tokens should be real, <15% padding)
    
    Strategy: Greedy bin packing
        1. Sort pending samples by n_tokens (descending)
        2. Greedily fill sequence until full
        3. Pad remaining space
    
    Ưu tiên: samples cùng resolution cho GPU efficiency
    (nhưng KHÔNG bắt buộc — khác resolution vẫn pack được)
    """
    
    def __init__(self, max_seq_len=4096, max_samples=32):
        self.L = max_seq_len
        self.max_samples = max_samples
    
    def pack(self, samples: list) -> dict:
        """
        Input:  list of converted 4D sample dicts
                mỗi cái có tokens(N_i, 1152), positions(N_i, 4), etc.
        
        Output: packed dict ready for model
        
        Algorithm:
            1. Sort samples by N (descending) cho greedy packing hiệu quả
            2. Lần lượt thêm samples cho đến khi không fit
            3. Pad phần còn lại
        """
        # Sort by token count (large first → fill gaps with small)
        samples_sorted = sorted(samples, key=lambda s: s["n_tokens"], reverse=True)
        
        packed_tokens = []
        packed_positions = []
        packed_freq_raw = []
        packed_freq_embed = []
        packed_sample_ids = []
        sample_metadata = []
        
        current_len = 0
        sample_id = 0
        
        for sample in samples_sorted:
            N = sample["n_tokens"]
            
            # Check: còn đủ chỗ không?
            if current_len + N > self.L:
                continue  # skip sample này, thử sample nhỏ hơn
            
            # Check: chưa quá nhiều samples?
            if sample_id >= self.max_samples:
                break
            
            # Add sample
            packed_tokens.append(sample["tokens"])
            packed_positions.append(sample["positions"])
            packed_freq_raw.append(sample["freq_raw"])
            packed_freq_embed.append(sample["freq_embed"])
            packed_sample_ids.append(
                torch.full((N,), sample_id, dtype=torch.long)
            )
            
            sample_metadata.append({
                "sample_id": sample_id,
                "modality": sample["modality"],
                "caption": sample["caption"],
                "n_tokens": N,
                "start_idx": current_len,
                "end_idx": current_len + N,
            })
            
            current_len += N
            sample_id += 1
        
        # ── Pad remaining ──
        remaining = self.L - current_len
        if remaining > 0:
            d_token = packed_tokens[0].shape[-1] if packed_tokens else 1152
            d_freq = packed_freq_raw[0].shape[-1] if packed_freq_raw else 15
            d_femb = packed_freq_embed[0].shape[-1] if packed_freq_embed else 128
            
            packed_tokens.append(torch.zeros(remaining, d_token))
            packed_positions.append(torch.zeros(remaining, 4))
            packed_freq_raw.append(torch.zeros(remaining, d_freq))
            packed_freq_embed.append(torch.zeros(remaining, d_femb))
            packed_sample_ids.append(
                torch.full((remaining,), -1, dtype=torch.long)  # -1 = padding
            )
        
        # ── Concatenate ──
        result = {
            "tokens":      torch.cat(packed_tokens, dim=0),      # (L, 1152)
            "positions":   torch.cat(packed_positions, dim=0),    # (L, 4)
            "freq_raw":    torch.cat(packed_freq_raw, dim=0),     # (L, 15)
            "freq_embed":  torch.cat(packed_freq_embed, dim=0),   # (L, 128)
            "sample_ids":  torch.cat(packed_sample_ids, dim=0),   # (L,)
            "n_samples":   sample_id,
            "n_real_tokens": current_len,
            "packing_efficiency": current_len / self.L,
            "sample_metadata": sample_metadata,
        }
        
        return result
    
    def pack_batch(self, all_samples: list, batch_size: int) -> list:
        """
        Pack nhiều sequences cho 1 batch.
        
        Input: nhiều converted samples
        Output: list of B packed dicts
        
        Strategy: distribute samples across B packs 
                  to maximize packing efficiency
        """
        # Shuffle samples
        random.shuffle(all_samples)
        
        packs = []
        remaining = list(all_samples)
        
        for _ in range(batch_size):
            if not remaining:
                break
            
            # Greedily select samples for this pack
            pack_samples = []
            pack_tokens = 0
            still_remaining = []
            
            for sample in remaining:
                if pack_tokens + sample["n_tokens"] <= self.L:
                    pack_samples.append(sample)
                    pack_tokens += sample["n_tokens"]
                else:
                    still_remaining.append(sample)
            
            remaining = still_remaining
            
            if pack_samples:
                packs.append(self.pack(pack_samples))
        
        return packs
```

### 4.2 Packing ví dụ

```
Ví dụ: L=4096, Stage 3 (all modalities)

Pending samples:
  img_256²  (256 tok) × 5
  img_512²  (1024 tok) × 2
  vid_8f256² (1024 tok) × 1
  3d_s32    (192 tok) × 1

Pack 1:
  ┌────────────┬──────────┬──────────┬────────┬────────┬──────┬───┐
  │vid_8f256²  │img_512²  │img_256²  │img_256²│img_256²│3d_s32│pad│
  │1024        │1024      │256       │256     │256     │192   │   │
  │id=0        │id=1      │id=2      │id=3    │id=4    │id=5  │   │
  └────────────┴──────────┴──────────┴────────┴────────┴──────┴───┘
  Total real: 3008 / 4096 = 73.4% efficiency
  Padding: 1088 tokens

Pack 2:
  ┌──────────┬──────────┬──────────┬──────────────────────────────┐
  │img_512²  │img_256²  │img_256²  │padding                      │
  │1024      │256       │256       │2560                          │
  │id=0      │id=1      │id=2      │                              │
  └──────────┴──────────┴──────────┴──────────────────────────────┘
  Total real: 1536 / 4096 = 37.5% efficiency (low — needs more samples)

Improvement: overfetch samples from datasets to have enough for good packing
```

---

## 5. Mask Generation

### 5.1 Attention Mask (Block-Diagonal)

```python
def build_attention_mask(sample_ids: Tensor) -> Tensor:
    """
    Block-diagonal mask: tokens CHỈ attend within same sample.
    
    sample_ids: (L,) — sample index, -1 = padding
    Returns: (L, L) bool — True = can attend
    
    Ví dụ sample_ids = [0,0,0, 1,1,1,1, 2,2, -1,-1]:
    
          0 0 0  1 1 1 1  2 2  - -
    0   [ 1 1 1  0 0 0 0  0 0  0 0 ]
    0   [ 1 1 1  0 0 0 0  0 0  0 0 ]
    0   [ 1 1 1  0 0 0 0  0 0  0 0 ]
    1   [ 0 0 0  1 1 1 1  0 0  0 0 ]
    1   [ 0 0 0  1 1 1 1  0 0  0 0 ]
    1   [ 0 0 0  1 1 1 1  0 0  0 0 ]
    1   [ 0 0 0  1 1 1 1  0 0  0 0 ]
    2   [ 0 0 0  0 0 0 0  1 1  0 0 ]
    2   [ 0 0 0  0 0 0 0  1 1  0 0 ]
    -   [ 0 0 0  0 0 0 0  0 0  0 0 ]
    -   [ 0 0 0  0 0 0 0  0 0  0 0 ]
    """
    valid = (sample_ids >= 0)  # (L,)
    same_sample = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1))  # (L, L)
    mask = same_sample & valid.unsqueeze(0) & valid.unsqueeze(1)
    return mask
```

### 5.2 Graph Mask (Per-Sample Subgraphs)

```python
def build_graph_mask(sample_ids: Tensor) -> Tensor:
    """
    Mask cho Module 3 (graph construction):
    Affinity giữa tokens khác sample = -inf → never neighbors.
    
    Returns: (L, L) float — 0 = same sample, -inf = different/padding
    """
    valid = (sample_ids >= 0)
    same_sample = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1))
    both_valid = valid.unsqueeze(0) & valid.unsqueeze(1)
    
    mask = torch.where(same_sample & both_valid, 0.0, float('-inf'))
    return mask
```

### 5.3 Mamba Scan Boundaries

```python
def build_mamba_boundaries(sample_ids: Tensor) -> list:
    """
    Mamba needs to scan PER SAMPLE, resetting state between samples.
    
    Returns: list of (start_idx, end_idx, scan_order) per sample
    
    Ví dụ sample_ids = [0,0,0, 1,1,1,1, 2,2, -1,-1]:
    → [(0, 3, order_0), (3, 7, order_1), (7, 9, order_2)]
    """
    boundaries = []
    current_id = -1
    start = 0
    
    for i in range(len(sample_ids)):
        sid = sample_ids[i].item()
        
        if sid < 0:  # padding
            if current_id >= 0:
                boundaries.append((start, i, current_id))
                current_id = -1
            continue
        
        if sid != current_id:
            if current_id >= 0:
                boundaries.append((start, i, current_id))
            start = i
            current_id = sid
    
    if current_id >= 0:
        boundaries.append((start, len(sample_ids), current_id))
    
    return boundaries
```

---

## 6. Progressive Training Stages

### 6.1 Stage Configuration

```python
STAGE_CONFIGS = {
    1: {
        "name": "Image Foundation",
        "steps": 200_000,
        "modalities": ["image"],
        "image_res_range": (64, 512),
        "video_res_range": None,
        "object3d_enabled": False,
        
        # Sampling (chỉ image)
        "task_weights": {
            "image_recon": 1.0,
        },
        
        # Freeze config
        "siglip2_freeze": "all",     # hoàn toàn frozen
        "stf_config": {
            "spatial_freq": True,     # DWT active
            "temporal_freq": False,   # STFT(t) disabled
            "depth_freq": False,      # STFT(z) disabled
        },
        
        # Graph config
        "graph_k": 12,
        "graph_spectral_k": 32,
        
        # Packing
        "max_seq_len": 4096,
        "batch_size": 8,     # packs per GPU
        
        # Optimizer
        "lr": 1e-4,
        "warmup_steps": 10_000,
        "weight_decay": 0.05,
        
        # Loss weights
        "loss_weights": {
            "recon": 1.0,
            "perceptual": 1.0,
            "gram": 1.0,
            "kl": 0.001,
            "vf_alignment": 0.0,     # chưa activate
            "understanding": 0.0,    # chưa activate
        },
    },
    
    2: {
        "name": "Video Dynamics",
        "steps": 200_000,
        "modalities": ["image", "video"],
        "image_res_range": (64, 1024),
        "video_res_range": (64, 512),
        "video_frame_range": (4, 16),
        "object3d_enabled": False,
        
        # Sampling (following AToken Table 2)
        "task_weights": {
            "image_recon": 0.222,      # Iʳ
            "video_understand": 0.111,  # Vᵘ
            "video_recon": 0.666,       # Vʳ (dominant!)
        },
        
        # Freeze config
        "siglip2_freeze": "except_last_4",  # unfreeze last 4 layers
        "stf_config": {
            "spatial_freq": True,
            "temporal_freq": True,      # STFT(t) NOW ACTIVE
            "depth_freq": False,
        },
        
        # Graph config (temporal edges now possible)
        "graph_k": 9,
        "graph_spectral_k": 32,
        
        # Packing
        "max_seq_len": 4096,
        "batch_size": 4,     # video takes more memory
        
        # Optimizer
        "lr": 5e-5,          # lower than stage 1 (finetuning)
        "warmup_steps": 5_000,
        "weight_decay": 0.05,
        
        # Loss weights
        "loss_weights": {
            "recon": 1.0,
            "perceptual": 1.0,
            "gram": 1.0,
            "kl": 0.001,
            "vf_alignment": 0.3,       # activate DINOv2 alignment
            "understanding": 0.1,      # start understanding loss
        },
        
        # Initialize from Stage 1 checkpoint
        "init_from": "stage1_best.pt",
    },
    
    3: {
        "name": "3D Geometry",
        "steps": 50_000,
        "modalities": ["image", "video", "3d"],
        "image_res_range": (64, 2048),
        "video_res_range": (64, 1024),
        "video_frame_range": (4, 32),
        "object3d_config": {"triplane_res": 32, "n_views": 8},
        
        # Sampling (following AToken Table 2)
        "task_weights": {
            "image_recon": 0.222,       # Iʳ
            "video_understand": 0.111,   # Vᵘ
            "video_recon": 0.444,        # Vʳ
            "3d_understand": 0.111,      # 3Dᵘ
            "3d_recon": 0.111,           # 3Dʳ
        },
        
        # Freeze config
        "siglip2_freeze": "none",   # fully unfrozen
        "stf_config": {
            "spatial_freq": True,
            "temporal_freq": True,
            "depth_freq": True,      # STFT(z) NOW ACTIVE
        },
        
        # Graph config
        "graph_k": 9,
        "graph_spectral_k": 32,
        
        # Packing
        "max_seq_len": 4096,
        "batch_size": 4,
        
        # Optimizer
        "lr": 3e-5,
        "warmup_steps": 2_000,
        "weight_decay": 0.05,
        
        # Loss weights
        "loss_weights": {
            "recon": 1.0,
            "perceptual": 1.0,
            "gram": 1.0,
            "kl": 0.001,
            "vf_alignment": 0.5,
            "understanding": 0.3,
        },
        
        # Initialize from Stage 2 checkpoint
        "init_from": "stage2_best.pt",
    },
}
```

### 6.2 What changes between stages — visual summary

```
╔══════════════════════════════════════════════════════════════════════╗
║                    STAGE TRANSITION MAP                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Stage 1 → Stage 2:                                                 ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │ ADDED:                                                         │  ║
║  │   ✓ VideoDataset activated                                    │  ║
║  │   ✓ STFT temporal activated in STF module                     │  ║
║  │   ✓ Temporal dimension in 4D positions (t > 0)                │  ║
║  │   ✓ Video samples in NaViT packing                            │  ║
║  │   ✓ Understanding loss (text alignment)                       │  ║
║  │   ✓ VF alignment loss (DINOv2)                                │  ║
║  │                                                                │  ║
║  │ CHANGED:                                                       │  ║
║  │   ○ SigLIP2: full freeze → last 4 layers unfrozen             │  ║
║  │   ○ LR: 1e-4 → 5e-5 (lower for finetuning)                   │  ║
║  │   ○ Image ratio: 100% → 22.2%                                 │  ║
║  │   ○ Image resolution: max 512 → max 1024                      │  ║
║  │   ○ Graph k: 12 → 9 (fewer neighbors, video tokens are many)  │  ║
║  │                                                                │  ║
║  │ UNCHANGED:                                                     │  ║
║  │   ● Model architecture (all modules same)                     │  ║
║  │   ● Packing mechanism (same NaViT)                            │  ║
║  │   ● Graph builder, spectral PE (same code, new data)          │  ║
║  │   ● MambaVision, SpectralGraphConv, Attention+Memory          │  ║
║  └────────────────────────────────────────────────────────────────┘  ║
║                                                                      ║
║  Stage 2 → Stage 3:                                                 ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │ ADDED:                                                         │  ║
║  │   ✓ Object3DDataset activated                                 │  ║
║  │   ✓ STFT depth activated in STF module                        │  ║
║  │   ✓ Depth dimension in 4D positions (z > 0)                   │  ║
║  │   ✓ 3D samples in NaViT packing                               │  ║
║  │   ✓ 3D reconstruction + understanding tasks                   │  ║
║  │   ✓ GaussianCube decoder head                                 │  ║
║  │                                                                │  ║
║  │ CHANGED:                                                       │  ║
║  │   ○ SigLIP2: last 4 layers → fully unfrozen                   │  ║
║  │   ○ LR: 5e-5 → 3e-5                                           │  ║
║  │   ○ Video recon ratio: 66.6% → 44.4%                          │  ║
║  │   ○ Image resolution: max 1024 → max 2048                     │  ║
║  │   ○ Video frame range: 4-16 → 4-32                            │  ║
║  │   ○ VF alignment weight: 0.3 → 0.5                            │  ║
║  │                                                                │  ║
║  │ UNCHANGED:                                                     │  ║
║  │   ● All architecture modules                                  │  ║
║  │   ● Packing mechanism                                         │  ║
║  │   ● Checkpoint loaded from stage 2 (continue training)        │  ║
║  └────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 6.3 Per-stage packing examples

```
═══ Stage 1 (Image only, L=4096) ═══

Pack: nhiều images khác resolution → maximize GPU utilization

  ┌────────┬────────┬────────┬──────────┬────────┬────────┬───────┬───┐
  │img 64² │img 64² │img 128²│img 256²  │img 256²│img 512²│img128²│pad│
  │16 tok  │16 tok  │64 tok  │256 tok   │256 tok │1024 tok│64 tok │   │
  └────────┴────────┴────────┴──────────┴────────┴────────┴───────┴───┘
  Tokens: 16+16+64+256+256+1024+64 = 1696 / 4096 = 41%
  
  Cải thiện: thêm images cho đến full
  ┌────────┬────────┬──────────┬──────────┬──────────┬────────┬────────┬───┐
  │img 512²│img 512²│img 256²  │img 256²  │img 256²  │img 256²│img 128²│pad│
  │1024    │1024    │256       │256       │256       │256     │64      │   │
  └────────┴────────┴──────────┴──────────┴──────────┴────────┴────────┴───┘
  Tokens: 1024+1024+256+256+256+256+64 = 3136 / 4096 = 76%  ← tốt hơn


═══ Stage 2 (Image + Video, L=4096) ═══

Pack mixes image và video tokens:

  ┌──────────────────┬──────────┬──────────┬──────────┬──────────┬───┐
  │video 8f×256²     │img 256²  │img 256²  │img 256²  │img 128²  │pad│
  │1024 tok          │256       │256       │256       │64        │   │
  │t=0..3, z=0       │t=0, z=0  │t=0, z=0  │t=0, z=0  │t=0, z=0  │   │
  └──────────────────┴──────────┴──────────┴──────────┴──────────┴───┘
  Tokens: 1024+256+256+256+64 = 1856 / 4096 = 45%

  Tốt hơn:
  ┌──────────────────┬──────────────────┬──────────┬──────────┬───┐
  │video 16f×256²    │video 8f×256²     │img 256²  │img 256²  │pad│
  │2048 tok          │1024 tok          │256       │256       │   │
  └──────────────────┴──────────────────┴──────────┴──────────┴───┘
  Tokens: 2048+1024+256+256 = 3584 / 4096 = 87%  ← excellent


═══ Stage 3 (Image + Video + 3D, L=4096) ═══

  ┌──────────────────┬─────────────────────────┬──────────┬───┐
  │video 8f×256²     │3D triplane S=64,p=4     │img 256²  │pad│
  │1024 tok          │768 tok                  │256       │   │
  │t=0..3, z=0       │t=0, x/y/z               │t=0, z=0  │   │
  └──────────────────┴─────────────────────────┴──────────┴───┘
  Tokens: 1024+768+256 = 2048 / 4096 = 50%
  
  Tốt hơn:
  ┌──────────────────┬──────────────────┬──────────┬──────────┬──────────┬───┐
  │video 16f×256²    │3D S=32,p=4      │img 512²  │img 256²  │img 256²  │pad│
  │2048              │192              │1024      │256       │256       │   │
  └──────────────────┴──────────────────┴──────────┴──────────┴──────────┴───┘
  Tokens: 2048+192+1024+256+256 = 3776 / 4096 = 92%  ← excellent
```

---

## 7. Optimizer & Scheduler

```python
class MAVTOptimizer:
    """
    Config per stage (values shown for Stage 1):
        optimizer:    AdamW
        lr:           1e-4
        betas:        (0.9, 0.95)
        weight_decay: 0.05
        warmup:       10k steps (linear warmup)
        schedule:     cosine decay to 1e-6
        grad_clip:    1.0
        mixed_prec:   bfloat16
    
    Parameter groups (different LR for different components):
        Group 1: SigLIP2 (frozen or very low LR)
          lr_mult = 0.0 (stage 1) → 0.1 (stage 2) → 0.3 (stage 3)
        
        Group 2: STF + Graph Builder + Spectral PE (lightweight, learn fast)
          lr_mult = 1.0
        
        Group 3: MambaVision stages 1-2 (CNN, stable)
          lr_mult = 1.0
        
        Group 4: MambaVision stages 3-4 (Mamba + Attention, need care)
          lr_mult = 0.5 (slightly lower to prevent instability)
        
        Group 5: Titans Memory (new module, needs warmup)
          lr_mult = 0.3 (stage 1) → 1.0 (stage 2+)
        
        Group 6: Decoder (largest, needs most compute)
          lr_mult = 1.0
        
        Group 7: Latent Projection
          lr_mult = 1.0
    """
    
    def __init__(self, model, stage_config):
        self.stage = stage_config
        
        # Build parameter groups
        param_groups = self._build_param_groups(model)
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=stage_config["lr"],
            betas=(0.9, 0.95),
            weight_decay=stage_config["weight_decay"],
        )
        
        # Scheduler: linear warmup + cosine decay
        self.scheduler = self._build_scheduler()
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')
    
    def _build_param_groups(self, model):
        groups = []
        
        # SigLIP2
        siglip_params = list(model.patchify.siglip2.parameters())
        freeze_mode = self.stage.get("siglip2_freeze", "all")
        
        if freeze_mode == "all":
            for p in siglip_params:
                p.requires_grad_(False)
        elif freeze_mode == "except_last_4":
            for p in siglip_params:
                p.requires_grad_(False)
            for p in model.patchify.siglip2.blocks[-4:].parameters():
                p.requires_grad_(True)
            groups.append({
                "params": list(model.patchify.siglip2.blocks[-4:].parameters()),
                "lr": self.stage["lr"] * 0.1,
                "name": "siglip2_last4",
            })
        elif freeze_mode == "none":
            groups.append({
                "params": siglip_params,
                "lr": self.stage["lr"] * 0.3,
                "name": "siglip2_full",
            })
        
        # STF + Graph + Spectral PE
        groups.append({
            "params": (
                list(model.stf.parameters()) +
                list(model.graph_builder.parameters()) +
                list(model.spectral_pe.parameters())
            ),
            "lr": self.stage["lr"] * 1.0,
            "name": "frequency_graph",
        })
        
        # Encoder
        groups.append({
            "params": list(model.encoder.parameters()),
            "lr": self.stage["lr"] * 0.8,
            "name": "encoder",
        })
        
        # Decoder (largest component)
        groups.append({
            "params": list(model.decoder.parameters()),
            "lr": self.stage["lr"] * 1.0,
            "name": "decoder",
        })
        
        # Latent projection
        groups.append({
            "params": list(model.latent_proj.parameters()),
            "lr": self.stage["lr"] * 1.0,
            "name": "latent_proj",
        })
        
        return groups
    
    def _build_scheduler(self):
        warmup = self.stage["warmup_steps"]
        total = self.stage["steps"]
        
        def lr_lambda(step):
            if step < warmup:
                return step / warmup  # linear warmup
            else:
                # cosine decay
                progress = (step - warmup) / (total - warmup)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
```

---

## 8. Complete Training Loop

```python
class MAVTTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.converter = Unified4DConverter(config)
        self.packer = NaViTTokenPacker(
            max_seq_len=config.max_seq_len,
            max_samples=config.max_samples_per_pack,
        )
        self.loss_fn = MAVTLoss(config)
    
    def train(self):
        """Full training across all stages."""
        for stage_idx in [1, 2, 3]:
            stage_config = STAGE_CONFIGS[stage_idx]
            print(f"\n{'='*60}")
            print(f"Starting Stage {stage_idx}: {stage_config['name']}")
            print(f"Steps: {stage_config['steps']}")
            print(f"Modalities: {stage_config['modalities']}")
            print(f"{'='*60}\n")
            
            # Load checkpoint from previous stage
            if "init_from" in stage_config:
                self.model.load_state_dict(
                    torch.load(stage_config["init_from"]), strict=False
                )
            
            # Setup optimizer for this stage
            optim = MAVTOptimizer(self.model, stage_config)
            
            # Setup datasets for this stage
            datasets = self._build_datasets(stage_config)
            sampler = ModalityWeightedSampler(
                datasets=datasets,
                weights=stage_config["task_weights"],
            )
            
            # Training loop
            self.model.train()
            for step in range(stage_config["steps"]):
                # ── Sample raw data ──
                raw_samples = sampler.sample(n=config.samples_per_step)
                
                # ── Convert to 4D tokens ──
                converted = [self.converter.convert(s) for s in raw_samples]
                
                # ── Pack into sequences ──
                packs = self.packer.pack_batch(
                    converted, batch_size=stage_config["batch_size"]
                )
                
                # ── Build masks ──
                batch = self._prepare_batch(packs)
                
                # ── Forward ──
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output = self.model(batch)
                    losses = self.loss_fn(
                        output, batch, stage_config["loss_weights"]
                    )
                
                # ── Backward ──
                optim.scaler.scale(losses["total"]).backward()
                
                # ── Gradient clip + step ──
                optim.scaler.unscale_(optim.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optim.scaler.step(optim.optimizer)
                optim.scaler.update()
                optim.scheduler.step()
                optim.optimizer.zero_grad()
                
                # ── Logging ──
                if step % 100 == 0:
                    self._log(step, stage_idx, losses, packs)
                
                # ── Validation ──
                if step % 5000 == 0:
                    metrics = self._validate(stage_config)
                    self._log_metrics(step, stage_idx, metrics)
                
                # ── Checkpoint ──
                if step % 10000 == 0:
                    self._save_checkpoint(stage_idx, step)
            
            # Save final checkpoint for this stage
            self._save_checkpoint(stage_idx, "final")
    
    def _prepare_batch(self, packs):
        """Stack packs into batch tensors + generate masks."""
        batch = {
            "tokens":     torch.stack([p["tokens"] for p in packs]).cuda(),
            "positions":  torch.stack([p["positions"] for p in packs]).cuda(),
            "freq_raw":   torch.stack([p["freq_raw"] for p in packs]).cuda(),
            "freq_embed": torch.stack([p["freq_embed"] for p in packs]).cuda(),
            "sample_ids": torch.stack([p["sample_ids"] for p in packs]).cuda(),
        }
        
        # Build masks
        B, L = batch["sample_ids"].shape
        batch["attn_mask"] = torch.stack([
            build_attention_mask(batch["sample_ids"][b])
            for b in range(B)
        ])  # (B, L, L)
        
        batch["graph_mask"] = torch.stack([
            build_graph_mask(batch["sample_ids"][b])
            for b in range(B)
        ])  # (B, L, L)
        
        batch["mamba_boundaries"] = [
            build_mamba_boundaries(batch["sample_ids"][b])
            for b in range(B)
        ]
        
        # Gather metadata
        batch["sample_metadata"] = [p["sample_metadata"] for p in packs]
        
        return batch
    
    def _log(self, step, stage, losses, packs):
        """Log training metrics."""
        avg_efficiency = np.mean([p["packing_efficiency"] for p in packs])
        
        wandb.log({
            "step": step,
            "stage": stage,
            "loss/total": losses["total"].item(),
            "loss/recon": losses.get("recon", 0),
            "loss/perceptual": losses.get("perceptual", 0),
            "loss/gram": losses.get("gram", 0),
            "loss/kl": losses.get("kl", 0),
            "loss/vf": losses.get("vf_alignment", 0),
            "loss/understand": losses.get("understanding", 0),
            "packing/efficiency": avg_efficiency,
            "packing/n_samples_avg": np.mean([p["n_samples"] for p in packs]),
            "lr": self.optimizer.param_groups[0]["lr"],
        })


class ModalityWeightedSampler:
    """
    Sample raw data from per-modality datasets
    according to task weights.
    
    Handles:
    - Different dataset sizes (image 100M vs 3D 800K)
    - Oversampling small datasets (3D repeated ~100×)
    - Task-specific sampling (recon vs understanding)
    """
    
    def __init__(self, datasets: dict, weights: dict):
        self.datasets = datasets
        self.weights = weights
        
        # Build sampling distribution
        self.tasks = list(weights.keys())
        self.probs = [weights[t] for t in self.tasks]
        
        # Map task → modality
        self.task_to_modality = {
            "image_recon": "image",
            "video_recon": "video",
            "video_understand": "video",
            "3d_recon": "3d",
            "3d_understand": "3d",
        }
        
        # Per-modality iterators
        self.iterators = {}
        for modality, ds in datasets.items():
            self.iterators[modality] = iter(DataLoader(
                ds, shuffle=True, num_workers=4, drop_last=True
            ))
    
    def sample(self, n: int) -> list:
        """Sample n raw data items according to task weights."""
        samples = []
        tasks = random.choices(self.tasks, weights=self.probs, k=n)
        
        for task in tasks:
            modality = self.task_to_modality[task]
            
            try:
                raw = next(self.iterators[modality])
            except StopIteration:
                self.iterators[modality] = iter(DataLoader(
                    self.datasets[modality], shuffle=True,
                    num_workers=4, drop_last=True
                ))
                raw = next(self.iterators[modality])
            
            # Add task info
            raw["task"] = task
            raw["modality"] = modality
            samples.append(raw)
        
        return samples
```

---

## 9. Compute Budget Estimation

```
╔══════════════════════════════════════════════════════════════╗
║ Hardware Assumption: 32× NVIDIA H100 80GB                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ Per-GPU budget:                                             ║
║   Memory: ~70GB usable (after framework overhead)           ║
║   BF16 throughput: ~1000 TFLOPS peak                        ║
║   Effective MFU: ~45% (realistic for training)              ║
║   → ~450 TFLOPS effective per GPU                           ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ Per-step computation:                                       ║
║   Forward:  ~60 GFLOPs per pack (L=4096, d=1152)           ║
║   Backward: ~120 GFLOPs (2× forward)                       ║
║   Total:    ~180 GFLOPs per pack per step                   ║
║   Batch=4 packs: ~720 GFLOPs per GPU per step              ║
║                                                              ║
║ Throughput:                                                  ║
║   450 TFLOPS / 720 GFLOPS = ~625 steps/second... (too fast) ║
║   Reality: memory bandwidth limited → ~3-5 steps/second     ║
║   Conservative: ~2 steps/second per GPU                     ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ Training time:                                               ║
║                                                              ║
║ Stage 1 (200k steps):                                       ║
║   200k / 2 steps_per_sec = 100k seconds = ~28 hours         ║
║   32 GPUs → still 28 hours (data parallel)                  ║
║   GPU hours: 32 × 28 = 896 H100-hours                      ║
║                                                              ║
║ Stage 2 (200k steps):                                       ║
║   Video larger → slower, ~1.5 steps/sec                     ║
║   200k / 1.5 = 133k sec = ~37 hours                        ║
║   GPU hours: 32 × 37 = 1,184 H100-hours                    ║
║                                                              ║
║ Stage 3 (50k steps):                                        ║
║   3D + high-res → ~1.0 steps/sec                           ║
║   50k / 1.0 = 50k sec = ~14 hours                          ║
║   GPU hours: 32 × 14 = 448 H100-hours                      ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║ TOTAL:                                                       ║
║   Wall time: ~79 hours ≈ 3.3 days                           ║
║   GPU hours: ~2,528 H100-hours                              ║
║                                                              ║
║ vs AToken: 138k H100-hours (256 GPUs × 22 days)            ║
║ MAVT: ~55× cheaper! (smaller model + linear backbone)       ║
║                                                              ║
║ Note: estimate conservative, actual may be 2-5× higher      ║
║ due to graph construction, spectral PE computation, I/O     ║
║ Realistic estimate: ~5k-10k H100-hours                      ║
╚══════════════════════════════════════════════════════════════╝
```
