# ViT/Patch Feature Supervision for Gaussian Splatting

## Owner

`src/gs_feature_supervision`

## Responsibility

This module provides a narrow feature-supervision boundary for Gaussian splatting experiments. It does not own dataset loading, COLMAP, full 3DGS training, checkpointing, or evaluation.

## Input

- Per-Gaussian 2D projected centers: `means_xy`, shape `[N, 2]`.
- Per-Gaussian projected log scales: `log_scales_xy`, shape `[N, 2]`.
- Per-Gaussian opacity logits: `opacity_logits`, shape `[N]`.
- Per-Gaussian values to render, such as patch features `h_i`, shape `[N, C]`.
- Optional depths for front-to-back ordering, shape `[N]`.
- Teacher patch feature map from ViT or a precomputed feature adapter, shape `[H_p, W_p, C]`.

## Output

- Rendered patch feature map, shape `[H_p, W_p, C]`.
- Accumulated alpha map, shape `[H_p, W_p]`.
- Feature supervision loss, usually `MSE(rendered_feature, teacher_feature)`.

## Main Flow

```text
training image
-> ViT or precomputed patch feature adapter
-> teacher patch feature map

Gaussian primitives with h_i
-> alpha_blend_splat_2d(...)
-> rendered patch feature map

rendered patch feature map + teacher patch feature map
-> feature_mse_loss
-> total training loss
```

## Dependency Direction

```text
experiment / training script
  -> src.gs_feature_supervision
  -> torch
```

The optional `TorchvisionVitPatchFeatureExtractor` is isolated in `vit_adapter.py` because model construction is an external dependency boundary.

## Error Semantics

- Shape mismatches raise `ValueError`.
- The renderer assumes projected 2D Gaussian parameters are already valid for the requested output grid.
- If a real 3DGS CUDA rasterizer exists, it should replace `alpha_blend_splat_2d` behind the same loss contract.

## Impact Scope

- Affects only new feature-supervision helpers and the toy experiment.
- Does not modify the existing paper notes, translated PDFs, SeaThru-NeRF code, or existing output checkpoints.

## Verification Method

Run the toy comparison:

```bash
/home/leo/miniconda3/bin/conda run -n sjtu python experiments/feature_splatting_toy.py --steps 300
```

Expected artifact:

```text
outputs/feature_splatting_toy/metrics.json
outputs/feature_splatting_toy/target.png
outputs/feature_splatting_toy/rgb_only_render.png
outputs/feature_splatting_toy/rgb_plus_feature_render.png
```

Current repository limitation: there is no complete 3DGS training entrypoint in this workspace. Therefore this verification proves the feature-rendering supervision path works, but it does not prove improvement on a full 3D reconstruction benchmark.

## Current Toy Results

The toy experiment uses sparse RGB supervision and patch-level teacher features to compare RGB-only optimization against RGB plus feature supervision.

With `--feature-weight 0.005`:

| Variant | Full RGB PSNR | Sparse RGB MSE | Feature MSE |
|---|---:|---:|---:|
| RGB only | 27.1722 | 0.001307 | 0.139333 |
| RGB + feature | 27.1684 | 0.001309 | 0.000494 |

With `--feature-weight 0.05`:

| Variant | Full RGB PSNR | Sparse RGB MSE | Feature MSE |
|---|---:|---:|---:|
| RGB only | 27.1722 | 0.001307 | 0.139333 |
| RGB + feature | 27.0216 | 0.001259 | 0.000252 |

Interpretation:

- The feature-rendering loss is active and strongly improves rendered-feature alignment.
- In this toy setup, RGB PSNR does not improve; a small weight is nearly neutral, while larger weights hurt full-image RGB PSNR.
- This supports using ViT/patch features as an auxiliary signal, but not as proof that it improves final NVS quality without a full 3DGS benchmark and careful weighting.
