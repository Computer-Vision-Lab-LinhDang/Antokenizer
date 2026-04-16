# AnToken

AnToken is a PyTorch Lightning implementation of a unified visual tokenizer
inspired by Apple's AToken work. This repo currently ships a practical MVP for:

- Stage 1: image reconstruction + semantic alignment
- Stage 2: image/video reconstruction with shared 4D tokenization
- Stage 4: optional discrete token heads (FSQ or multi-codebook VQ)

3D support is intentionally stubbed and deferred, matching the scope decision in
`antoken.md`.

## Quick Start

```bash
pip install -e ".[dev]"
antoken-train fit --config configs/base.yaml --config configs/stages/stage1_image.yaml
```

## Layout

- `src/antoken/`: model, data, losses, utils
- `configs/`: LightningCLI yaml configs
- `scripts/`: train, eval, visualization entrypoints
- `tests/`: focused shape and utility tests
