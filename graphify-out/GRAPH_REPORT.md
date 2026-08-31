# Graph Report - /private/tmp/claude/mavtgraph  (2026-08-19)

## Corpus Check
- Corpus is ~29,605 words - fits in a single context window. You may not need a graph.

## Summary
- 392 nodes · 694 edges · 19 communities detected
- Extraction: 70% EXTRACTED · 30% INFERRED · 0% AMBIGUOUS · INFERRED: 210 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Data loading & datasets|Data loading & datasets]]
- [[_COMMUNITY_Research direction & extensions|Research direction & extensions]]
- [[_COMMUNITY_Encoder & decoder blocks|Encoder & decoder blocks]]
- [[_COMMUNITY_Loss & training objective|Loss & training objective]]
- [[_COMMUNITY_MAVT model core|MAVT model core]]
- [[_COMMUNITY_Architecture concepts & DDT|Architecture concepts & DDT]]
- [[_COMMUNITY_Lightning training module|Lightning training module]]
- [[_COMMUNITY_Model forward integration|Model forward integration]]
- [[_COMMUNITY_Patchify & 4D positions|Patchify & 4D positions]]
- [[_COMMUNITY_Content-detail split|Content-detail split]]
- [[_COMMUNITY_Evaluation metrics|Evaluation metrics]]
- [[_COMMUNITY_Latent projection heads|Latent projection heads]]
- [[_COMMUNITY_Isolated node|Isolated node]]
- [[_COMMUNITY_Isolated node|Isolated node]]
- [[_COMMUNITY_Isolated node|Isolated node]]
- [[_COMMUNITY_Isolated node|Isolated node]]
- [[_COMMUNITY_Isolated node|Isolated node]]
- [[_COMMUNITY_Isolated node|Isolated node]]
- [[_COMMUNITY_Isolated node|Isolated node]]

## God Nodes (most connected - your core abstractions)
1. `StandardTransformerBlock` - 31 edges
2. `MAVT` - 26 edges
3. `MAVTLightningModule` - 19 edges
4. `HybridBackbone` - 19 edges
5. `PatchifyEncoder` - 17 edges
6. `MAVTLoss` - 16 edges
7. `ContentDetailSplit` - 16 edges
8. `AsymmetricDecoder` - 15 edges
9. `MAVTDataModule` - 15 edges
10. `UnderstandingDecoder` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Content-Dynamics Split (keyframe + delta)` --semantically_similar_to--> `Content-Detail Split (Slot attn + window pooling)`  [INFERRED] [semantically similar]
  plan/DDT_Implementation_Plan.md → docs/MAVT_Report.md
- `VoxelRGAT (3D Chebyshev typed-edge attention)` --semantically_similar_to--> `Removed module3_graph (O(N^2) OOM)`  [INFERRED] [semantically similar]
  docs/MAVT_Report.md → plan/DDT_Implementation_Plan.md
- `Removed Stubs (ChebyshevGraphConv, MambaVisionMixer, TitansMemory)` --semantically_similar_to--> `Titans Memory (removed in MAVT v3)`  [INFERRED] [semantically similar]
  plan/DDT_Implementation_Plan.md → docs/MAVT_Report.md
- `Shared 4D RoPE (core/rope4d.py)` --semantically_similar_to--> `4D Rotary Position Embedding (rope4d.py)`  [INFERRED] [semantically similar]
  plan/DDT_Implementation_Plan.md → docs/model.md
- `Content-Dynamics Split (keyframe + delta)` --semantically_similar_to--> `Unified Content-Detail Split (Slot Attn + DynamicsPooler)`  [INFERRED] [semantically similar]
  plan/DDT_Implementation_Plan.md → docs/model.md

## Hyperedges (group relationships)
- **MAVT modality-agnostic tokenizer pipeline** — model_patchify, model_encoder, model_latent, model_decoder, model_cd_split [INFERRED 0.85]
- **D2O three-component cache optimization** — research_variance_alloc, research_token_eviction, research_cross_scale_merging [EXTRACTED 0.90]
- **DDT 3-level frequency integration (enrich + select + loss)** — ddt_wavelet_enrich, ddt_dwt_selection, ddt_subband_loss [EXTRACTED 0.90]
- **MAVT 7-stage forward pass flow** — patchify_PatchifyEncoder, backbone_HybridBackbone, content_detail_split_ContentDetailSplit, latent_heads_VAEHead, decoder_AsymmetricDecoder [INFERRED 0.90]
- **Pre-LN attention+MLP residual block protocol** — transformer_StandardTransformerBlock, rgat_RGAT4DBlock, decoder_WindowedSelfAttn2D, content_detail_split_CrossAttentionLayer [INFERRED 0.75]
- **Dual heads decoding shared VAE latent z** — latent_heads_VAEHead, decoder_AsymmetricDecoder, decoder_UnderstandingDecoder [INFERRED 0.80]
- **L_total = recon(L1+LPIPS) + KL + CLIP + distill + slot-div + temporal, EMA-weighted** — losses_MAVTLoss, losses_LPIPSLoss, losses_cosine_distill_loss, losses_ModalityEMAWeighter [INFERRED 0.90]
- **train-step flow: MAVT model forward -> MAVTLoss -> AdamW optimizer (Lightning)** — lightning_module_MAVTLightningModule, mavt_MAVT, losses_MAVTLoss [INFERRED 0.85]
- **data pipeline: datasets -> _collate -> ModalityGroupedBatchSampler -> DataModule** — datamodule_MAVTDataModule, datamodule__collate, datamodule_ModalityGroupedBatchSampler [INFERRED 0.80]

## Communities

### Community 0 - "Data loading & datasets"
Cohesion: 0.08
Nodes (24): _collate(), _dist_info(), MAVTDataModule, ModalityGroupedBatchSampler, PyTorch Lightning DataModule for MAVT multi-modal training., DataModule supporting 3-stage curriculum.      Stage 1: image only     Stage 2:, Collate a batch — all items must share the same modality., Batch sampler for ConcatDataset that keeps each batch single-modality.      Indi (+16 more)

### Community 1 - "Research direction & extensions"
Cohesion: 0.06
Nodes (48): D2O Final Comparison vs Baselines, D2OOptimizer Module (variance + eviction + merge), Related KV-Cache Work (H2O, FastGen), D2O Correct Positioning, Rationale: D2O is Optimization Layer, Not Architecture, Tokenizer Related Work (VQVAE, VQGAN), MAVT Extension Report (Vietnamese), MAVT 7-Stage Pipeline (+40 more)

### Community 2 - "Encoder & decoder blocks"
Cohesion: 0.07
Nodes (24): _copy_siglip2_block(), Stage 2: Hybrid Transformer-RGAT4D Backbone.  Block layout (12 blocks):   0-3:, Load SigLIP2 backbone weights into Transformer blocks (best-effort).          fr, Best-effort copy from a SigLIP2 encoder layer to our StandardTransformerBlock., 12-block hybrid Transformer-RGAT backbone., FourDQueryEncoding, PixelShuffleCNNDecoder, z               : (B, N_c+N_d, latent_dim)         target_positions: (N_target, (+16 more)

### Community 3 - "Loss & training objective"
Cohesion: 0.07
Nodes (28): MAVTDataModule (multi-modal curriculum), ModalityGroupedBatchSampler (single-modality batches), _collate (single-modality batch collate), ShardVideoDataset (video2dataset shards), SyntheticMultiModalDataset (smoke-test stub), UniversalImageDataset, UniversalThreeDDataset (triplane PNG), UniversalVideoDataset (+20 more)

### Community 4 - "MAVT model core"
Cohesion: 0.12
Nodes (23): HybridBackbone, ContentDetailSplit, Eagerly create content poolers for a known (N_c, N_d) combo.          Call once, Content-Detail Split module.      Separates tokens into a content channel (seman, AsymmetricDecoder, Stage 5: Two decoder heads on the shared latent z.    AsymmetricDecoder     — z, Full asymmetric decoder: expander → self-attention blocks → CNN upsample.      H, Decode a single 2D grid → (B, 3, H_out, W_out). (+15 more)

### Community 5 - "Architecture concepts & DDT"
Cohesion: 0.07
Nodes (39): 3D Triplane Decoupling (XY=content, XZ/YZ=dynamics), Content-Dynamics Split (keyframe + delta), DDT: Dual-Domain Tokenizer, DDT Dual-Domain Tokenizer Implementation Plan, Dual-Domain Token (f_semantic SigLIP2 + f_freq DWT), DWT Energy Top-K Selection (O(N)), FA-VAE Insight (detail-weighted subband loss), MAVT Refactor to Clean Baseline (42% LOC cut) (+31 more)

### Community 6 - "Lightning training module"
Cohesion: 0.1
Nodes (15): _make_teacher_input(), MAVTLightningModule, PyTorch Lightning Module for MAVT training.  Supports 3-stage curriculum via `tr, Read active modality + resolution from the attached DataModule and         eager, Use DataModule as single source of truth for active_modalities.          Overrid, Load frozen SigLIP2 vision tower as teacher for cosine distillation., Keep frozen teacher in eval mode regardless of train()/eval() calls., Strip frozen teacher weights from checkpoints to keep them small. (+7 more)

### Community 7 - "Model forward integration"
Cohesion: 0.16
Nodes (22): HybridBackbone, ContentDetailSplit, CrossAttentionLayer, SlotPooler, AsymmetricDecoder, FourDQueryEncoding, PixelShuffleCNNDecoder, ResBlock2D (+14 more)

### Community 8 - "Patchify & 4D positions"
Cohesion: 0.13
Nodes (12): FourDPositionEmbedding, _image_positions(), PatchifyEncoder, Stage 1: Unified Conv3d patchification for image, video, and 3D triplane inputs., Apply Conv3d to a single-frame input with causal padding.          x: (B, 3, H,, x: (B, 3, H, W) → tokens (B, N, D), positions (N, 4), plane_ids (N,), Learned 4D position embedding for (t, x, y, z) coordinates., x: (B, 3, T, H, W) → tokens (B, N, D), positions (N, 4), plane_ids (N,) (+4 more)

### Community 9 - "Content-detail split"
Cohesion: 0.13
Nodes (10): _compute_metrics(), CrossAttentionLayer, _default_positions(), Stage 3: Content-Detail Split via slot cross-attention.  ContentExtractor   → N_, Pool residual tokens in local coordinate windows.          Returns         -----, Single cross-attention + FFN layer (pre-LN)., Returns         -------         compressed : (B, N_c + N_d_local, D)         met, Slot cross-attention pooler: learns to pool N tokens into num_slots tokens. (+2 more)

### Community 10 - "Evaluation metrics"
Cohesion: 0.27
Nodes (11): compute_image_metrics(), compute_threed_metrics(), compute_video_metrics(), psnr(), Evaluation metrics for MAVT.  Image  : rFID (requires torchmetrics-image), PSNR,, PSNR in dB. Inputs expected in [0, 1]., Simplified SSIM via torchmetrics if available, else MSE proxy., pred, target: (B, 3, H, W) in [0, 1]. (+3 more)

### Community 11 - "Latent projection heads"
Cohesion: 0.22
Nodes (4): Stage 4: Dual Latent Projection heads.  VAEHead      → μ, logσ², z  (reparameter, Attention-pooled semantic projection: [C;D] tokens → 768-d sequence vector., compressed : (B, N, D) → s : (B, out_dim), SemanticHead

### Community 12 - "Isolated node"
Cohesion: 1.0
Nodes (0): 

### Community 13 - "Isolated node"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Isolated node"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Isolated node"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Isolated node"
Cohesion: 1.0
Nodes (1): Fallback positions for direct unit tests without patch metadata.

### Community 17 - "Isolated node"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Isolated node"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **74 isolated node(s):** `Stage 6: Loss functions for MAVT training.  L_total = w_recon·(w_l1·L1 + w_lpips`, `pred, target: (B, 3, H, W) or (B, 3, N, H, W) in [-1, 1] or [0, 1].          5-D`, `1 - mean cosine similarity. Teacher is detached (no gradient flows back).`, `Penalise high cosine similarity between content slots (collapse prevention).`, `Match frame-to-frame motion between pred and target.      pred, target: (B, 3, T` (+69 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Isolated node`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Isolated node`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Isolated node`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Isolated node`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Isolated node`** (1 nodes): `Fallback positions for direct unit tests without patch metadata.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Isolated node`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Isolated node`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MAVT` connect `MAVT model core` to `Patchify & 4D positions`, `Lightning training module`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `MAVTLoss` connect `Lightning training module` to `Loss & training objective`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `StandardTransformerBlock` connect `Encoder & decoder blocks` to `MAVT model core`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `StandardTransformerBlock` (e.g. with `FourDQueryEncoding` and `UnifiedDetailExpander`) actually correct?**
  _`StandardTransformerBlock` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `MAVT` (e.g. with `MAVTLightningModule` and `PyTorch Lightning Module for MAVT training.  Supports 3-stage curriculum via `tr`) actually correct?**
  _`MAVT` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `MAVTLightningModule` (e.g. with `MAVT` and `MAVTLoss`) actually correct?**
  _`MAVTLightningModule` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `HybridBackbone` (e.g. with `StandardTransformerBlock` and `RGAT4DBlock`) actually correct?**
  _`HybridBackbone` has 11 INFERRED edges - model-reasoned connections that need verification._