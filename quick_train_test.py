#!/usr/bin/env python3
"""Quick end-to-end training sanity check."""
import torch
import lightning as L
from mavt.training.lightning_module import MAVTLightningModule
from mavt.data.datamodule import MAVTDataModule

torch.set_float32_matmul_precision('medium')

module = MAVTLightningModule(
    embed_dim=64, num_heads=4, num_blocks=4,
    latent_dim=8, dec_dim=32, num_dec_attn_blocks=2,
    semantic_dim=32, mlp_ratio=2.0,
    use_gradient_checkpointing=False,
    use_lpips=False, training_stage=1,
    warmup_steps=1, total_steps=5,
)
dm = MAVTDataModule(
    active_modalities=['image'], image_resolution=64,
    batch_size=2, num_workers=0, synthetic_n=8,
)
trainer = L.Trainer(
    max_steps=3, enable_checkpointing=False,
    logger=False, enable_progress_bar=True,
    accelerator='cpu',
)
trainer.fit(module, dm)
print('Training loop OK')
