# LOSS+DENSIFY
lpw002 = train-lpips-weight 0.02
lpw = LPIPS weight
# ours_denstify LPIPS Loss 复现实验

本文记录 `ours_denstify` 当前用于追 LPIPS 指标的训练改动和服务器端复现实验入口。

## LPIPS Loss 是怎么算的

原始训练监督只优化 3DGS 常见的 L1 + DSSIM：

$$
L_{\mathrm{photo}}
=
(1-\lambda_{\mathrm{dssim}})\,L_1(I, G)
+
\lambda_{\mathrm{dssim}}\,
\frac{1-\mathrm{SSIM}(I,G)}{2}
$$

其中：

- \(I\)：当前相机视角下的渲染图；
- \(G\)：对应 GT 图；
- \(\lambda_{\mathrm{dssim}}\)：`--lambda-dssim`，默认 `0.2`。

新增的训练目标是在指定步数后加入可反传的 SeaSplat VGG LPIPS：

$$
L
=
L_{\mathrm{photo}}
+
w_{\mathrm{lpips}}\,
L_{\mathrm{lpips}}(I_{s}, G_{s})
$$

其中：

- \(w_{\mathrm{lpips}}\)：`--train-lpips-weight`；
- \(I_s, G_s\)：把 \(I,G\) 按 `--train-lpips-max-size` 下采样后的图像；
- \(L_{\mathrm{lpips}}\)：`methods/seasplat/lpipsPyTorch` 中的 VGG LPIPS，和当前 `seasplat + vgg` 测评协议一致。

实现边界：

- `--patch-perceptual-weight` 只影响 patch-guided densification 的选点打分，不参与 loss 反传。
- `--train-lpips-weight` 才是真正进入训练 loss 的 LPIPS 权重。
- 训练 LPIPS 从 `--train-lpips-start-step` 开始启用，避免早期几何不稳定时直接优化感知特征。
- 训练时可用 `--train-lpips-max-size 512` 控制显存和速度；最终测评仍对保存的 full-resolution PNG 用 `evaluate_lpips_image_pairs.py` 复算。

## 新增接口

Owner：`methods/ours_denstify/train_3dgs_scene.py`

Responsibility：训练 3DGS，并在需要时把可微 LPIPS loss 加到主训练目标里。

Input：

- `--train-lpips-weight`：LPIPS loss 权重，默认 `0.0`，表示关闭。
- `--train-lpips-start-step`：开始启用 LPIPS loss 的 step，默认 `7000`。
- `--train-lpips-max-size`：训练 LPIPS 的最长边，默认 `512`；设为 `0` 表示不缩放。
- `--train-lpips-net`：LPIPS backbone，默认 `vgg`。
- `--train-lpips-backend`：训练 LPIPS backend，目前只支持 `seasplat`。

Output：

- `training_summary.json` 会写入 `train_lpips_weight`、`train_lpips_start_step`、`train_lpips_max_size`、`train_lpips_net`、`train_lpips_backend`。
- `benchmark_patch_densification.py` 会把这些字段透传到训练脚本，并写入 `summary.csv`。

Error semantics：

- 非 `seasplat` 的 `--train-lpips-backend` 会直接报错。
- 渲染图和 GT 形状不一致会直接报错，不做隐式 resize 对齐。

Dependency direction：

```text
benchmark_patch_densification.py
  -> train_3dgs_scene.py
  -> methods/ours_denstify/lpips_backend.py
  -> methods/seasplat/lpipsPyTorch
```

## 服务器端复现命令

先同步当前本地代码到服务器独立目录：

```bash
rsync -azR --info=progress2 \
  -e "ssh -p 30896 -o StrictHostKeyChecking=no" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  methods/ours_denstify modules utils methods/semantic_importance.py methods/seasplat/lpipsPyTorch \
  root@sc01-ssh.gpuhome.cc:/root/rivermind-data/SJTU/ours_denstify_current_repo/
```

单场景 IUI3 复现实验，`lpw002` 对应 `--train-lpips-weight 0.02`：

```bash
cd /root/rivermind-data/SJTU/ours_denstify_current_repo

/home/user/micromamba/bin/python3 methods/ours_denstify/benchmark_patch_densification.py \
  --data-root /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset \
  --scenes IUI3-RedSea/undistorted_pinhole \
  --out /root/rivermind-data/SJTU/outputs/ours_denstify_iui3_train_lpips_lpw002_20260614 \
  --variants patch_guided \
  --seeds 0 \
  --iterations 19999 \
  --factor -1 \
  --holdout 8 \
  --holdout-offset 0 \
  --train-eval-every 1000 \
  --lpips \
  --lpips-net vgg \
  --lpips-backend seasplat \
  --patch-size 32 \
  --patch-detail-lambda 3.0 \
  --patch-edge-weight 0.75 \
  --patch-perceptual-weight 0.5 \
  --patch-perceptual-max-size 768 \
  --train-lpips-weight 0.02 \
  --train-lpips-start-step 7000 \
  --train-lpips-max-size 512 \
  --train-lpips-net vgg \
  --train-lpips-backend seasplat \
  --densify-grad-threshold 2e-6 \
  --log-every 500 \
  --compact-summary
```

final PNG 复算 LPIPS：

```bash
/home/user/micromamba/bin/python3 methods/ours_denstify/evaluate_lpips_image_pairs.py \
  --pairs-dir /root/rivermind-data/SJTU/outputs/ours_denstify_iui3_train_lpips_lpw002_20260614/undistorted_pinhole/patch_guided/seed_000/test_renders \
  --lpips-net vgg \
  --lpips-backend seasplat \
  --out-csv /root/rivermind-data/SJTU/outputs/ours_denstify_iui3_train_lpips_lpw002_20260614/undistorted_pinhole/patch_guided/seed_000/test_renders/lpips_remote_seasplat_image_pairs.csv
```

best LPIPS checkpoint 复算：

```bash
BEST_CKPT=/root/rivermind-data/SJTU/outputs/ours_denstify_iui3_train_lpips_lpw002_20260614/undistorted_pinhole/patch_guided/seed_000/train/best/best_lpips.ply
BEST_OUT=/root/rivermind-data/SJTU/outputs/ours_denstify_iui3_train_lpips_lpw002_20260614/undistorted_pinhole/patch_guided/seed_000/test_renders_best_lpips

/home/user/micromamba/bin/python3 methods/ours_denstify/render_3dgs_views.py \
  --data /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/IUI3-RedSea/undistorted_pinhole \
  --checkpoint "$BEST_CKPT" \
  --out "$BEST_OUT" \
  --split test \
  --factor -1 \
  --holdout 8 \
  --holdout-offset 0 \
  --lpips \
  --lpips-net vgg \
  --lpips-backend seasplat

/home/user/micromamba/bin/python3 methods/ours_denstify/evaluate_lpips_image_pairs.py \
  --pairs-dir "$BEST_OUT" \
  --lpips-net vgg \
  --lpips-backend seasplat \
  --out-csv "$BEST_OUT/lpips_remote_seasplat_image_pairs.csv"
```

## 当前已验证结果

IUI3 pilot `lpw002`：

| checkpoint | PSNR | SSIM | benchmark LPIPS | PNG 复算 LPIPS |
|---|---:|---:|---:|---:|
| final / best_lpips | `29.513597` | `0.898566` | `0.199613` | `0.198212` |

对比 SeaSplat IUI3 LPIPS `0.2029`，当前 `lpw002` 的 PNG 复算 LPIPS 已经更低。

## Verification

本地语法检查：

```bash
/home/leo/miniconda3/bin/conda run -n sfquant python -m compileall -q methods/ours_denstify modules
```

服务器参数检查：

```bash
cd /root/rivermind-data/SJTU/ours_denstify_current_repo
/home/user/micromamba/bin/python3 methods/ours_denstify/train_3dgs_scene.py --help | grep train-lpips
/home/user/micromamba/bin/python3 methods/ours_denstify/benchmark_patch_densification.py --help | grep train-lpips
```
