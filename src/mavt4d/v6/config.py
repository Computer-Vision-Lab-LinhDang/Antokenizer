"""v6 architecture: shared trunk over evidence + state, explicit Kalman D, sparse box-limited field."""
from dataclasses import asdict, dataclass

# Encoder layouts that pretrained weights can be loaded into (encoder.load_backbone).
BACKBONES = {
    'dinov3_vits16': dict(encoder_width=384, encoder_heads=6, encoder_layers=12, encoder_registers=4),
    'siglip2_b16': dict(encoder_width=768, encoder_heads=12, encoder_layers=12, encoder_registers=0),
    'siglip2_so400m16': dict(encoder_width=1152, encoder_heads=16, encoder_layers=27, encoder_registers=0,
                             encoder_mlp_hidden=4304),
}


@dataclass(frozen=True)
class V6Config:
    # State
    slots: int = 24
    latent_width: int = 128          # C_obj token, C_evt, h, background widths
    c_obj_tokens: int = 6
    anchors: int = 32
    anchors_per_token: int = 8       # anchors / anchors_per_token D_neo tokens per slot
    descriptor_dim: int = 16
    background_tokens: int = 16
    max_cameras: int = 7
    camera_rows: int = 32            # packed per-camera calibration row, see contracts.CameraRow
    # Trunk
    trunk_width: int = 384
    trunk_layers: int = 12
    trunk_heads: int = 6
    ownership_layers: tuple = (4, 8, 12)
    logit_scale_max: float = 30.0
    # Encoder (DINOv3 ViT-S/16 layout by default; see BACKBONES)
    backbone: str = 'dinov3_vits16'
    image_height: int = 192
    image_width: int = 256
    patch_size: int = 16
    tubelet: int = 2
    encoder_width: int = 384
    encoder_layers: int = 12
    encoder_heads: int = 6
    encoder_registers: int = 4
    encoder_mlp_hidden: int = 0       # 0: 4 x encoder_width
    depth_hypotheses: int = 4
    fourier_bands: int = 6
    # Write rules
    lambda_max: float = 64.0          # cap on accumulated evidence Lambda per anchor
    visibility_min: float = 0.1       # s_min of the hard C write mask
    depth_conf_min: float = 0.1       # rho_min of the hard C write mask
    rgb_only_lambda_cap: float = 0.25 # lambda ceiling for sources without measured depth
    min_anchors_for_shape: int = 3    # fewer measured anchors: no C_obj shape write
    log_r_floor: float = -9.0         # measurement log-variance floor (m^2)
    q_min: float = 1e-4               # process noise floor
    event_chi2_99: float = 11.34      # chi2_3 at 0.99, per 3-axis group
    tentative_steps: int = 2
    miss_steps: int = 2
    # Decoder
    samples_per_segment: int = 8
    max_slots_per_ray: int = 2
    box_sigma_inflation: float = 2.0
    box_soft_margin: float = 0.25
    decoder_width: int = 256
    schema_version: int = 6

    def __post_init__(self):
        if self.anchors % self.anchors_per_token:
            raise ValueError('anchors must split evenly into D_neo tokens')
        if self.trunk_width % self.trunk_heads or self.encoder_width % self.encoder_heads:
            raise ValueError('width must divide by heads')
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError('image size must be a multiple of the patch size')
        if self.max_cameras < 1 or self.slots < 1 or self.depth_hypotheses < 1:
            raise ValueError('invalid camera, slot or depth-hypothesis count')
        if self.backbone not in BACKBONES:
            raise ValueError(f'unknown backbone {self.backbone!r}')
        if self.schema_version != 6:
            raise ValueError('v6 config requires schema_version 6')

    @classmethod
    def with_backbone(cls, name: str, **overrides):
        """Config whose encoder shape matches a pretrained backbone; explicit overrides win."""
        if name not in BACKBONES:
            raise ValueError(f'unknown backbone {name!r}')
        return cls(**{'backbone': name, **BACKBONES[name], **overrides})

    @property
    def neo_tokens(self):
        return self.anchors // self.anchors_per_token

    @property
    def tokens_per_slot(self):
        # 6 C_obj + 1 C_evt + 1 D_pose + D_neo + 1 h
        return self.c_obj_tokens + 3 + self.neo_tokens

    @property
    def scene_tokens(self):
        # one token per camera row, background tokens, one global token
        return self.max_cameras + self.background_tokens + 1

    @property
    def patches_per_camera(self):
        return (self.image_height // self.patch_size) * (self.image_width // self.patch_size)

    def to_dict(self):
        return asdict(self)
