tôi đang implement một bộ unified tokenizer cho image, video, 3d object. Lấy ý tưởng từ AToken (Apple)

Kiến trúc sẽ như sau: 
__________________________________________________________________________
   |         Unified tokenizer: Image + Video + 3D -> Shared 4D latent space  |
   |__________________________________________________________________________|
            |                      |                         |
    ________v________      ________v________      ___________v___________
   |      Image      |    |      Video      |    |       3D object       |
   | t=0, z=0, (x,y) |    | z=0, (t,x,y) seq|    | t=0, (x,y,z) voxel    |
   | Patchify -> Toks|    | Space-time blocks|    | Multi-view -> SLAT    |
   |_________________|    |_________________|    |_______________________|
            |                      |                         |
            └----------------------o-------------------------┘
                                   |
                     [ Linear projection + 4D RoPE ]
                                   |
                ___________________v___________________
               |          Shared ViT encoder           |
               | Pretrained SigLIP2 + 4D RoPE | SA     |
               |_______________________________________|
                                   |
                                   v  z ∈ ℝ N×D
     ______________________        ________________        ________________
    |  Continuous latent   |      | Discrete tokens|      |Semantic features|
    | For diffusion/flow   |      | For AR Gen     |      | For understanding|
    | AE, KL reg           |      | Multi-codebook |      | CLIP/SigLIP align|
    |______________________|      |________________|      |________________|
               |                           |                       |
               └---------------------------o-----------------------┘
                                           |
                                   [ z_q hoặc z_c ]
                ___________________________v___________________________
               |               Modality-specific decoder heads         |
               | Image -> RGB | Video -> Frames | 3D -> GSplats/mesh   |
               |_______________________________________________________|
                                           |
      _____________________________________v____________________________________
     |                 |                   |                   |                |
 ____v____        _____v____          _____v____          _____v_____      _____v_____
|  Image  |      |  Video   |        |    3D    |        |  Under-   |    |           |
|  recon  |      |  recon   |        |  recon   |        |  standing |    |   LOSS    |
| L1+LPIPS|      | L1+Optic |        | Rend+LPIP|        | CLIP loss |    |  FUNCTIONS|
|_________|      |__________|        |__________|        |___________|    |___________|

    __________________________________________________________________________
   |                     PROGRESSIVE TRAINING CURRICULUM                      |
   |__________________________________________________________________________|
    | Stage 1: Image  | Stage 2: + Video | Stage 3: + 3D    | Stage 4: Tokens |
    | 300K steps      | 200K steps       | 150K steps       | 100K steps      |
    |_________________|__________________|__________________|_________________|

Key insight #1: 4D latent space unifies everything
AToken là unified visual tokenizer đầu tiên đạt cả high-fidelity reconstruction lẫn semantic understanding across images, videos, và 3D assets, bằng cách encode tất cả vào một shared 4D latent space. Apple Machine Learning Research
Ý tưởng core cực kỳ elegant: mọi visual input đều có thể biểu diễn trong không gian 4 chiều (t, x, y, z):
Images map thành một single 2D (x,y) slice với t = z = 0. Videos thêm coordinate cho time (t, x, y, với z = 0). 3D objects map vào (x, y, z) grid với t = 0. DeepLearning.AI
Khi tất cả tokens đều mang 4D coordinates, một single transformer có thể process chúng qua self-attention — không cần architecture riêng cho mỗi modality. 4D RoPE encode relative position, nên model tự biết spatial/temporal/depth relationships.

Key insight #2: Adversarial-free training
AToken sử dụng adversarial-free training objective kết hợp perceptual và Gram matrix losses, đạt SOTA reconstruction quality mà không cần GAN — loại bỏ hoàn toàn instabilities. Terrencekim
Đây là simplification lớn. Thay vì train discriminator riêng (gây mode collapse, training oscillation), AToken chỉ dùng:

L1 reconstruction (pixel-level)
LPIPS perceptual (VGG features)
Gram matrix loss (texture statistics matching)
CLIP contrastive (semantic alignment)

Và vẫn beat VQGAN + GAN ở rFID. Với unified 3-modality training, bỏ GAN giảm complexity đáng kể.

Key insight #3: Progressive curriculum training
Model được train qua 4-stage progressive curriculum, dần dần thêm capabilities cho images, videos, và 3D, cùng optional stage cuối cho discrete tokenization. Terrencekim
Đây là chiến lược training cốt lõi:
Stage 1 → Image. Encoder mở rộng pretrained SigLIP2 từ 2D images sang 4D, generalize patch embedding thành space-time blocks t×p×p, với zero-initialized temporal weights giữ nguyên image features ban đầu. ResearchGate Bạn train reconstruction + semantic alignment trên image data trước.
Stage 2 → Add Video. Unfreeze temporal weights. AToken mở rộng latent dimensions từ 32 lên 48 để accommodate motion complexity. ResearchGate Dùng temporal tiling (16-32 frames → 4-8 latent frames) với KV-caching. Một finding quan trọng: multimodal training enhances single-modality performance — image reconstruction cải thiện khi thêm video và 3D. Terrencekim
Stage 3 → Add 3D. AToken mở rộng TRELLIS-SLAT cho 3D. ResearchGate 3D objects rendered thành multi-view images → encode vào shared latent → decode ra Gaussian Splats. Training trên Objaverse (800K 3D assets).
Stage 4 → Discrete tokens. Thêm quantization layer (FSQ hoặc multi-codebook) cho AR generation compatibility. Không degrade continuous performance.

Key insight #4: Modality-specific decoders, shared encoder
Encoder là shared — một ViT process mọi modality. Nhưng decoder cần modality-specific heads vì output formats khác nhau:

Image/Video head: Decode latent → RGB pixels
3D head: Decode ra Gaussian splats — small colored 3D blobs mà khi rendered cùng nhau tạo thành coherent 3D shape. DeepLearning.AI

- Trong đó từng phần sẽ như sau :

# Patchify:

 ## IMAGE PATCHIFY — nền tảng đơn giản nhất
AToken sử dụng unified patchification scheme cho tất cả modalities. Với input x ∈ ℝ^(T×H×W×3), chia thành non-overlapping space-time patches kích thước t_p × p × p. Với images (T=1), áp dụng temporal patch size 1. ResearchGate
Cơ chế cụ thể cho image:
Input: Ảnh 256×256×3 (RGB)
Bước 1 — Chia patches: Grid non-overlapping p×p patches (p=16 là chuẩn). Ảnh 256×256 → grid 16×16 = 256 patches. Mỗi patch chứa 16×16×3 = 768 pixels.
Bước 2 — Linear projection: Flatten mỗi patch thành vector 768-dim, rồi project lên D-dim embedding qua single linear layer: Linear(768, D). Với SigLIP2-So, D=1152.
Bước 3 — Gán 4D coordinates: Mỗi patch tại grid position (row=i, col=j) nhận tọa độ p = (t=0, x=i, y=j, z=0). Image là 2D slice trong 4D space — t và z cố định bằng 0.
Bước 4 — 4D RoPE: Trong mỗi attention layer, query/key vectors chia thành 4 groups bằng nhau. Mỗi group xoay (rotate) theo frequency tương ứng với 1 chiều (t, x, y, z). Vì t=z=0 cố định cho images, các groups đó không xoay → attention degenerates thành standard 2D spatial attention.

```python
# Pseudocode: Image patchify
def patchify_image(image, patch_size=16):
    """image: (B, 3, H, W) → tokens: (B, N, D), positions: (B, N, 4)"""
    B, C, H, W = image.shape
    h_patches, w_patches = H // patch_size, W // patch_size
    
    # Reshape thành patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.reshape(B, -1, C * patch_size * patch_size)  # (B, N, 768)
    
    # Linear projection  
    tokens = self.patch_embed(patches)  # (B, N, D)
    
    # 4D coordinates: (t=0, x=row, y=col, z=0)
    positions = []
    for i in range(h_patches):
        for j in range(w_patches):
            positions.append([0, i, j, 0])  # t=0, z=0
    positions = torch.tensor(positions)  # (N, 4)
    
    return tokens, positions
```

## VIDEO SPACE-TIME BLOCKS — mở rộng temporal
AToken generalize patch embedding thành space-time blocks kích thước t_p × p × p, với zero-initialized temporal weights giữ nguyên image features ban đầu. ResearchGate
Input: Video 16 frames × 256×256×3
Bước 1 — Space-time blocking: Thay vì patch 2D (p×p), dùng block 3D (t_p × p × p). VD: t_p=2, p=16 → mỗi block cover 2 frames × 16×16 pixels. Video 16f×256×256 → (16/2) × (256/16) × (256/16) = 8 × 16 × 16 = 2048 tokens.
Bước 2 — Conv3d patch embedding (inflated): Thay Conv2d(3, D, p×p, stride=p) bằng Conv3d(3, D, t_p×p×p, stride=(t_p, p, p)). Inflation trick:
python# Inflate 2D patch embed → 3D
W_3d = torch.zeros(D, 3, t_p, p, p)
W_3d[:, :, -1, :, :] = W_2d  # last-slice: pretrained spatial weights
W_3d[:, :, :-1, :, :] = 0    # zero temporal weights

Khi T=1 (image input): causal padding thêm (t_p-1) zero frames trước
Conv3d chỉ "thấy" frame thật ở slice cuối → output = output của Conv2d

Bước 3 — 4D coordinates (t, x, y, z=0): Mỗi block tại temporal index k, spatial (i,j) nhận p = (k, i, j, 0). z vẫn = 0 vì video là sequence of 2D frames.
Bước 4 — Causal temporal attention: Trong self-attention, thêm mask: tokens ở time t chỉ attend tokens ở time ≤ t. Spatial attention trong cùng frame vẫn full bidirectional. Điều này cho phép autoregressive frame prediction và variable-length inference.
Bước 5 — Temporal tiling cho video dài: Video dài hơn training window (VD: 128 frames vs 32-frame training) → chia tiles, process tuần tự. Key trick: KV-caching — tile sau reuse Key/Value tensors từ tiles trước, không cần recompute. Giữ temporal coherence across tiles.

```python
def patchify_video(video, temporal_patch=2, spatial_patch=16):
    """video: (B, 3, T, H, W) → tokens, positions"""
    B, C, T, H, W = video.shape
    t_blocks = T // temporal_patch
    h_blocks = H // spatial_patch  
    w_blocks = W // spatial_patch
    
    # Conv3d patch embedding (stride = patch size → non-overlapping)
    tokens = self.st_patch_embed(video)  # Conv3d, output: (B, D, t_blocks, h_blocks, w_blocks)
    tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)
    
    # 4D coordinates: (t=frame_group, x=row, y=col, z=0)
    positions = []
    for t in range(t_blocks):
        for i in range(h_blocks):
            for j in range(w_blocks):
                positions.append([t, i, j, 0])
    
    return tokens, torch.tensor(positions)
```

## 3D MULTI-VIEW VOXEL AGGREGATION — phức tạp nhất
Đây là phần khó nhất, kết hợp rendering, 2D feature extraction, và 3D voxel aggregation. AToken adapt TRELLIS-SLAT bằng cách render multi-view images từ spherically sampled cameras, apply unified patchification, rồi aggregate features vào voxel space. ResearchGate
TRELLIS-SLAT define local latents trên active voxels intersecting bề mặt object. Features được encode bằng cách fuse và process image features từ densely rendered views, extracted bởi pretrained DINOv2 encoder. arXiv
Pipeline cụ thể 5 bước:
Bước 1 — Multi-view rendering
Đặt N cameras (VD: 8-24) trên sphere xung quanh 3D object. Render mỗi view thành ảnh RGB R×R (VD: 512×512). Camera positions sampled đều trên sphere (elevation ±30°, azimuth 0°-360°).
python# Render N views from sphere
cameras = sample_cameras_on_sphere(N=8, radius=2.0, elevations=[-20, 20])
views = []
for cam in cameras:
    rgb_image = differentiable_render(mesh, cam, resolution=512)
    views.append(rgb_image)  # each: (3, 512, 512)
Bước 2 — Extract 2D features per view
Mỗi rendered view qua pretrained vision encoder (DINOv2 hoặc SigLIP2) → feature map. Các feature maps được extract bằng pretrained DINOv2 encoder từ randomly sampled camera views trên sphere. arXiv
python# Extract features from each view
feature_maps = []
for view in views:
    patches = patchify(view, patch_size=16)  # (1024, 768)
    features = encoder(patches)               # (1024, D) = (32×32, D) 
    feature_maps.append(features.reshape(32, 32, D))
Bước 3 — Xác định active voxels
Tạo 3D voxel grid V×V×V (VD: 64³ = 262,144 voxels). Chỉ giữ active voxels — voxels mà surface của object đi qua. Thường chỉ 5-15% total voxels là active (~10K-30K tokens). Xác định bằng cách check occupancy từ mesh hoặc point cloud.
python# Voxelize: find active voxels
voxel_grid = create_grid(resolution=64)  # 64×64×64
active_mask = check_surface_intersection(mesh, voxel_grid)
active_voxels = voxel_grid[active_mask]  # ~15K voxels
Bước 4 — Project & aggregate features vào voxels
Đây là bước core. Với mỗi active voxel, project 3D position lên tất cả N camera views, lấy feature tại projected (u,v) coordinate qua bilinear interpolation, rồi average:
Mỗi voxel được project lên multiview feature maps để retrieve features tại corresponding locations, và average của chúng được sử dụng. Grid 64³ là đủ để reconstruct 3D asset ở high fidelity nhờ representation capabilities mạnh của DINOv2 features. arXiv
python# Aggregate features into active voxels
for voxel in active_voxels:
    projected_feats = []
    for view_idx, cam in enumerate(cameras):
        # Project 3D voxel center → 2D pixel coordinate
        u, v = project_3d_to_2d(voxel.xyz, cam.intrinsic, cam.extrinsic)
        
        # Check visibility (not occluded by other geometry)
        if is_visible(voxel.xyz, cam, depth_buffer[view_idx]):
            # Bilinear sample feature at (u, v)
            feat = bilinear_sample(feature_maps[view_idx], u, v)  # (D,)
            projected_feats.append(feat)
    
    # Average across all visible views
    voxel.feature = torch.stack(projected_feats).mean(dim=0)  # (D,)
Bước 5 — Gán 4D coordinates (0, x, y, z)
Mỗi active voxel tại grid position (i, j, k) nhận tọa độ p = (t=0, i, j, k). t=0 vì 3D objects không có temporal dimension. Chỉ active voxels có tokens → sparse representation.
Sparse representation unify tất cả modalities bằng cách chỉ activate relevant dimensions: images occupy (x,y) plane tại t=z=0, videos extend along temporal axis với z=0, và 3D assets là surface voxels trong (x,y,z) space với t=0. ResearchGate
python# 4D coordinates for 3D
positions = []
for voxel in active_voxels:
    i, j, k = voxel.grid_index
    positions.append([0, i, j, k])  # t=0, (x,y,z) from voxel grid

# Encoder
Xây dựng VIT Encoder -> CÓ thể config 
Key insight: dùng pretrained SigLIP2 làm backbone, mở rộng sang 4D.

Modifications:
1. Generalize patch embedding: 2D p×p → 4D t×p×p×1, zero-init temporal weights
2. Thêm 4D Rotary Position Embedding (RoPE) vào mọi attention layer
3. Self-attention across ALL tokens bất kể modality

# Latent

Từ encoder Trả ra 2 nhánh latent là Contiguous / Discrete tokens cho reconstruct task và Semantic features cho understanding task 

# Decoder 
Decoder specific head cho từng task:

Reconstruct: Specific head cho từng modality : image, video, 3d object. 
Understanding: Head để train bằng clip và classification head 

# Losses 

Image  |      |  Video   |        |    3D    |        |  Under-   |   
|  recon  |      |  recon   |        |  recon   |        |  standing |
| L1+LPIPS|      | L1+Optic |        | Rend+LPIP|        | CLIP loss |
|_________|      |__________|        |__________|        |___________|

# Training

## Data
Bộ data có cấu trúc thư mục như sau:
```txt
.
├── 3d_objects
│   └── renders
│       ├── 00f980c0932a42af9340b2b8ce2b7915
│       │   ├── cameras.json
│       │   ├── view_0.png
│       │   ├── view_1.png
│       │   ├── view_2.png
│       │   ├── view_3.png
│       │   ├── view_4.png
│       │   ├── view_5.png
│       │   ├── view_6.png
│       │   └── view_7.png
├── captions
│   ├── 3d.json
│   ├── images.json
│   └── videos.json
├── images
│   ├── 04a0ca0e83b26aad.jpg
│   ├── 04a8312018949d65.jpg
│   ├── 0610309638eb76bc.jpg
│   ├── 0748d6903c3029c0.jpg
│   ├── 075287bdfae93425.jpg
│   ├── 075cd223c847ab34.jpg
│   ├── 0be9f11cadf609fa.jpg
└── videos
    ├── 1006973647.mp4 
    ├── 1007601001.mp4
    ├── 10087169.mp4
    ├── 1009947179.mp4 
```
Trong đó nội dung của các file json như sau:
cameras.json
```json
{"view_0": [[0.45671454071998596, 0.03144685551524162, 0.8890573382377625, -1.7781145572662354], [-0.8896133303642273, 0.01614448055624962, 0.4564290940761566, -0.912858247756958], [-1.0730173727324654e-07, -0.99937504529953, 0.035348959267139435, -0.07069803029298782], [0.0, 0.0, 0.0, 1.0]], "camera_angle_x": 0.6911112070083618, "view_1": [[-0.5988979935646057, 0.2877187728881836, -0.7473547458648682, 1.4947093725204468], [0.800825297832489, 0.2151709794998169, -0.5589098930358887, 1.1178196668624878], [1.7867920121261704e-07, -0.9332305788993835, -0.3592779338359833, 0.7185559868812561], [0.0, 0.0, 0.0, 1.0]], "view_2": [[0.4823670983314514, 0.844833493232727, 0.2314700186252594, -0.4629400670528412], [-0.8759691715240479, 0.46522173285484314, 0.12746277451515198, -0.2549256980419159], [-6.248116335427767e-08, -0.2642444372177124, 0.9644557237625122, -1.9289114475250244], [0.0, 0.0, 0.0, 1.0]], "view_3": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], "view_4": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], "view_5": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], "view_6": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], "view_7": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}
```

3d.json
```json
{
  "c3481b9029b449eab72fd0ec9bcdc6c6": "A stylized robot figure composed of blocky shapes in blue and orange, featuring a large rectangular head with circular eyes, a split torso, spiked arm attachments, long cylindrical legs, and orange feet.",
  "20498de9ab564bd690da58b4272811f6": "A rectangular white clipboard with a beige triangular clasp.",
  "707bb26b10424f45a1b6060011851f68": "A complex architectural roof structure with multiple pitched sections and a prominent gable supported by wooden beams.",
  "af498dd101544789a222eb1a8b6552f6": "A wooden chest filled with gold coins, jewelry, and stacked dice.",
}
```

images.json
```
{
"35f109c2aed39d21": "A man standing on top of a chair",
}
```

videos.json
```json
{
  "1052565473": "Group of people cycling on a trail next to a river. ",
  "33662158": "Someone opens a safe with weapons and money 4k, close-up",
}
```

# Implementation Goal
Cần phải cài đặt:
- Các lib cần thiết sử dụng
- Core model, losses, datamodule 
- Code training, Code evaluation, visualization 
- Code training yêu cầu sử dụng pytorch lightning, có log, save checkpoint eval đầy đủ, ddp 
- Cần phải config được kiến trúc mô hình bằng file yaml, có lightning cli để config dễ dàng 


