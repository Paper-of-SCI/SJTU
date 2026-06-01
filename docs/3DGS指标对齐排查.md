# 3DGS 指标对齐排查记录

## Owner

- Owner: `methods/3dgs` 实验入口与 `utils/dataset_loaders.py` 数据加载边界。
- 责任范围: 让本仓库的 SeaThru-NeRF 3DGS 对比实验在数据预处理、分辨率、split、LPIPS 口径上可复现、可解释。

## Responsibility

本文件记录为什么本仓库早期结果与 SWAGSplatting 论文中的 3DGS baseline 差异很大，以及后续实验必须使用的对齐协议。

不负责:

- 声称完全复现 SWAGSplatting 的私有训练配置。
- 替代真实主实验结论。
- 修改 patch-guided densification 的算法含义。

## Input

- 原始数据: `src/datasets/SeathruNeRF_dataset/<scene>`
- 对齐后数据: `outputs/aligned_datasets/SeathruNeRF_undistorted/<scene>`
- 训练入口: `methods/3dgs/benchmark_patch_densification.py`
- 单场景训练入口: `methods/3dgs/train_3dgs_scene.py`
- 渲染/指标入口: `methods/3dgs/render_3dgs_views.py`
- 预处理入口: `scripts/prepare_colmap_undistorted_dataset.py`
- 协议检查入口: `scripts/inspect_3dgs_protocol.py`
- SWAGS 论文数据包验证场景: `outputs/swags_dataset/new_dataset_v2/cur`

## Output

- 对齐后的 COLMAP 数据目录:

```text
outputs/aligned_datasets/SeathruNeRF_undistorted/
  Curasao/
    images/
    sparse/0/cameras.bin
    sparse/0/images.bin
    sparse/0/points3D.bin
  IUI3-RedSea/
  JapaneseGradens-RedSea/
  Panama/
```

- benchmark 输出:

```text
outputs/<experiment_name>/summary.csv
outputs/<experiment_name>/summary.json
outputs/<experiment_name>/<scene>/<variant>/seed_xxx/test_renders/metrics.csv
```

## Main Flow

1. 用 COLMAP `image_undistorter` 把原始 `OPENCV` 畸变相机转换成 undistorted `PINHOLE` 数据。
2. benchmark 使用对齐后的 `--data-root outputs/aligned_datasets/SeathruNeRF_undistorted`。
3. 分辨率使用 `--target-width 720`，对齐官方 3DGS/SWAGS 代码中的 `-r 720` 语义。
4. split 使用 `--holdout 8 --holdout-offset <offset>`，必须显式记录 offset。
5. LPIPS 使用 `--lpips-net vgg --lpips-backend official_3dgs`，对齐 graphdeco 官方 `metrics.py` 的 `lpipsPyTorch` 口径。

## Dependency Direction

```text
scripts/prepare_colmap_undistorted_dataset.py
  -> external COLMAP binary
  -> writes aligned dataset

scripts/inspect_3dgs_protocol.py
  -> utils/colmap_reader.py
  -> stdout only

methods/3dgs/benchmark_patch_densification.py
  -> train_3dgs_scene.py
  -> render_3dgs_views.py
  -> utils/dataset_loaders.py

utils/dataset_loaders.py
  -> utils/colmap_reader.py
  -> utils/image_utils.py
```

训练和渲染都只通过 `utils/dataset_loaders.py` 读取 split、分辨率和图像路径，避免训练与测试 split 分叉。

## Error Semantics

- 找不到输入图像目录或 `sparse/0`: 预处理脚本直接抛 `FileNotFoundError`。
- 输出目录已存在但没有 `--overwrite`: 预处理脚本抛 `FileExistsError`，避免覆盖已用数据。
- 同时传 `--target-height` 和 `--target-width`: loader 抛 `ValueError`。
- `patch_guided_semantic` 未传 `--semantic-importance-root`: benchmark 抛 `ValueError`。
- 缺少语义 mask: semantic provider 直接报错，避免静默退化成无语义版本。

## Impact Scope

影响:

- SeaThru-NeRF 3DGS baseline 与 patch-guided 系列实验。
- `summary.csv/json` 中新增/记录 `target_width`、`holdout_offset`、`lpips_backend` 等字段。
- 服务器上建议统一使用 `/root/SJTU/outputs/aligned_datasets/SeathruNeRF_undistorted`。

不影响:

- Gaussian 参数结构。
- RGB loss。
- renderer 输出格式。
- 已有默认行为: `holdout_offset=0`，不传 `target-width` 时仍按 `factor`。
- 默认 `densify_grad_threshold` 已从早期实验用的 `2e-5` 调整为 `2e-6`，用于对齐本仓库 gsplat 训练路径下的 official 3DGS Gaussian 数量级。旧实验可显式传 `--densify-grad-threshold 2e-5` 复现。

## Root Causes

### 1. 原始数据是畸变 `OPENCV` 相机

本地 Curasao 原始相机为:

```text
OPENCV 1776x1182
fx=1960.0787 fy=1961.7874 cx=894.3797 cy=580.6337
k1=-0.0853 k2=0.1475 p1=-0.00007 p2=0.00055
```

graphdeco 官方 3DGS 和 SWAGSplatting 的 COLMAP reader 默认只接受 `PINHOLE/SIMPLE_PINHOLE`。直接把 `OPENCV` 图像按 pinhole 跑，会在边缘视角产生明显误差。

COLMAP undistort 后 Curasao 变为:

```text
PINHOLE 1794x1188
```

四个场景已生成的 aligned 数据:

```text
Curasao                 PINHOLE 1794x1188
IUI3-RedSea             PINHOLE 1383x917
JapaneseGradens-RedSea  PINHOLE 1384x918
Panama                  PINHOLE 1795x1188
```

### 2. LPIPS 口径不同

早期结果使用 `lpips` package，数值会明显偏大。官方 3DGS/SWAGS 使用 vendored `lpipsPyTorch`，并在 `metrics.py` 中使用:

```python
lpips(render, gt, net_type="vgg")
```

因此后续 benchmark 必须使用:

```bash
--lpips --lpips-net vgg --lpips-backend official_3dgs
```

### 3. `720p` 口径要显式固定

官方 3DGS 的 `-r 720` 表示把图像宽度缩放到 720，而不是高度 720。本仓库使用:

```bash
--target-width 720
```

不要混用 `--target-height 720`，否则分辨率和论文/官方命令语义不一致。

### 4. `-r 720` 对应宽度 720，不是高度 720

official 3DGS 在 SWAGS `cur` 上使用 `-r 720` 后，render/gt 输出尺寸是:

```text
720x405
```

因此本仓库必须用:

```bash
--target-width 720
```

如果使用 `--target-height 720`，实际训练/测试分辨率会更大，指标不能和论文表格直接比较。

### 5. split 和具体预处理数据包对 Curasao 指标影响很大

默认 `--holdout 8 --holdout-offset 0` 的 SWAGS `cur` test 是:

```text
MTN_1288.jpg
MTN_1296.jpg
MTN_1304.jpg
```

其中 `MTN_1288.jpg` 是明显困难视角。official 3DGS + SWAGS `cur` + width720 + 20k 的 per-view 指标:

```text
MTN_1288: PSNR=20.99 SSIM=0.824 LPIPS=0.277
MTN_1296: PSNR=32.09 SSIM=0.961 LPIPS=0.094
MTN_1304: PSNR=34.43 SSIM=0.944 LPIPS=0.128
```

所以 held-out 图像本身会显著影响均值。SWAGS 论文数据包 `new_dataset_v2.zip` 中的 `cur` 是:

```text
PINHOLE 1281x721
images=21
points3D=23033
```

它不是本地原始 `OPENCV 1776x1182` 数据，也不是用本地 `images_wb` 临时 undistort 出来的 `PINHOLE 1794x1188` 数据。

### 6. gsplat 版 densification 梯度阈值需要校准

早期本仓库使用 `--densify-grad-threshold 2e-5`，20k 后只有约 `22k` 个 Gaussian；official 3DGS 在同一 SWAGS `cur` 协议下约 `189k` 个 Gaussian。容量不足会让细节和 LPIPS 明显偏差。

在本仓库 gsplat 训练路径下，`2e-6` 能把 standard baseline 校准到 official 数量级:

```text
ours standard_3dgs, SWAGS cur, width720, 20k:
Gaussian=196661
PSNR=29.5765 SSIM=0.9174 LPIPS=0.1813
```

因此后续对齐论文口径的实验默认使用 `2e-6`。如果需要复现早期实验，显式传 `--densify-grad-threshold 2e-5`。

## Evidence

Curasao，width 720，20,000 iterations，official VGG LPIPS:

| 数据协议 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| 原始 `OPENCV` 数据，忽略畸变硬跑 | 27.4520 | 0.8203 | 0.2024 |
| COLMAP undistorted `PINHOLE`，`holdout_offset=0` | 28.0187 | 0.8918 | 0.1636 |
| SWAGSplatting 论文表中 3DGS Curacao | 29.0234 | 0.9127 | 0.1790 |
| official 3DGS + SWAGS `cur` 数据包 | 29.1695 | 0.9098 | 0.1663 |
| 本仓库 standard + SWAGS `cur` + `2e-6` 阈值 | 29.5765 | 0.9174 | 0.1813 |

结论:

- 大 LPIPS 差距主要由 LPIPS backend、数据包/相机模型、`720` 分辨率语义、densification 阈值共同造成。
- 使用 SWAGS `cur` 数据包、`--target-width 720`、`--lpips-backend official_3dgs`、`--densify-grad-threshold 2e-6` 后，本仓库 baseline 已进入论文/official 量级。
- 本仓库 trainer 仍不是 graphdeco 官方源码；论文中若声明复现 official 3DGS baseline，必须明确训练入口和数据协议。

## External References Checked

- graphdeco official 3DGS `metrics.py`: 使用 `utils.loss_utils.ssim`、`utils.image_utils.psnr`、`lpipsPyTorch.lpips(..., net_type="vgg")`。
- SWAGSplatting GitHub `scene/dataset_readers.py`: 只接受 `PINHOLE/SIMPLE_PINHOLE`，并从 non-interpolated frames 中按 `idx % llffhold == 0` 选择 test frames。
- SWAGSplatting GitHub `metrics.py`: 用 `lpipsPyTorch` VGG，并在计算前把 render 用 `Image.NEAREST` resize 到 GT 尺寸。
- SWAGSplatting README: 论文数据包包含必要预处理数据，最终结构包含 `distorted/`、undistorted `images/`、`sparse/`、`depthmap/`、`masked/` 等目录。
- SWAGSplatting PDF: 写明 SeaThru-NeRF 每场景三张 validation，所有图像 resize 到 720p，每个模型训练 20,000 iterations。

## Rebuild Aligned Dataset

本地:

```bash
/home/leo/miniconda3/envs/sfquant/bin/python scripts/prepare_colmap_undistorted_dataset.py \
  --data-root src/datasets/SeathruNeRF_dataset \
  --out-root outputs/aligned_datasets/SeathruNeRF_undistorted \
  --overwrite
```

服务器上已同步到:

```text
/root/SJTU/outputs/aligned_datasets/SeathruNeRF_undistorted
```

检查相机模型:

```bash
cd /root/SJTU
python scripts/inspect_3dgs_protocol.py \
  outputs/swags_dataset/new_dataset_v2/cur \
  --target-width 720 \
  --holdout 8 \
  --holdout-offset 0
```

期望输出:

```text
scene: outputs/swags_dataset/new_dataset_v2/cur
images: total=21 train=18 test=3
test_images: MTN_1288.jpg, MTN_1296.jpg, MTN_1304.jpg
camera: id=1 model=PINHOLE source=1281x721 target=720x405 ...
```

## Recommended Benchmark Command

主对比建议使用:

```bash
cd /root/SJTU

python methods/3dgs/benchmark_patch_densification.py \
  --data-root outputs/aligned_datasets/SeathruNeRF_undistorted \
  --scenes Curasao IUI3-RedSea JapaneseGradens-RedSea Panama \
  --variants standard_3dgs patch_guided patch_guided_semantic patch_reallocate \
  --seeds 0 1 \
  --iterations 19999 \
  --factor 1 \
  --target-width 720 \
  --holdout 8 \
  --holdout-offset 0 \
  --lpips \
  --lpips-net vgg \
  --lpips-backend official_3dgs \
  --densify-grad-threshold 2e-6 \
  --patch-size 32 \
  --patch-detail-lambda 2.0 \
  --patch-edge-weight 0.75 \
  --semantic-importance-root semantic_importance \
  --semantic-base 0.2 \
  --reallocate-fraction 0.10 \
  --out outputs/3dgs_aligned_undistorted_720w_19999_all
```

如果只做 smoke:

```bash
cd /root/SJTU

python methods/3dgs/benchmark_patch_densification.py \
  --data-root outputs/aligned_datasets/SeathruNeRF_undistorted \
  --scenes Curasao \
  --variants standard_3dgs patch_guided \
  --seeds 0 \
  --iterations 1000 \
  --factor 1 \
  --target-width 720 \
  --holdout 8 \
  --holdout-offset 0 \
  --lpips \
  --lpips-net vgg \
  --lpips-backend official_3dgs \
  --densify-grad-threshold 2e-6 \
  --patch-size 32 \
  --out outputs/3dgs_aligned_undistorted_720w_smoke
```

## Verification Method

本地代码检查:

```bash
/home/leo/miniconda3/envs/sfquant/bin/python -m unittest tests.test_dataset_loaders tests.test_3dgs_patch_densification
/home/leo/miniconda3/envs/sfquant/bin/python -m py_compile \
  utils/dataset_loaders.py \
  methods/3dgs/train_3dgs_scene.py \
  methods/3dgs/render_3dgs_views.py \
  methods/3dgs/benchmark_patch_densification.py \
  scripts/prepare_colmap_undistorted_dataset.py \
  scripts/inspect_3dgs_protocol.py
```

服务器 smoke 已验证:

```text
data-root=outputs/aligned_datasets/SeathruNeRF_undistorted
scene=Curasao
variant=standard_3dgs
iterations=1000
target-width=720
lpips-backend=official_3dgs
holdout-offset=0
```

输出:

```text
summary.csv generated
final.ply generated
test_renders/metrics.csv generated
```
