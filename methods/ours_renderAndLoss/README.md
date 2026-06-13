# methods/ours 用法说明

## 1. 模块定位

**Owner**：`methods/ours`

**Responsibility**：提供当前 ours 方法的训练、渲染、指标评估和批量 benchmark 入口。

当前方法默认启用 3D medium field。主渲染形式是：

$$
I(u)
=
I_{\mathrm{obj}}(u)
+
I_{\mathrm{med}}(u)
$$

代码中对应：

- `rgb_object`：物体直接光项；
- `rgb_medium`：水体介质散射项；
- `pred_image` / `render.image`：最终图像；
- `medium.pt`：和 `.ply` 配套保存的 medium field checkpoint。

## 2. 依赖边界

`methods/ours` 只负责编排训练、渲染和保存，不直接实现 medium field 细节。

依赖方向：

```text
methods/ours/*.py
  -> modules/*
  -> utils/*
```

主要边界：

- medium 网络实现：`modules/medium_field.py`
- medium 渲染包装：`modules/medium_renderer.py`
- loss：`modules/losses.py`
- 3DGS 模型和 renderer：`modules/gaussian_model.py`、`modules/renderer.py`
- 数据加载：`utils/dataset_loaders.py`
- PLY 读写：`utils/ply_io.py`

## 3. 输入输出

训练入口：

```bash
methods/ours/train_3dgs_scene.py
```

输入：

- `--data`：COLMAP / SeathruNeRF 场景目录；
- `--out`：训练输出目录；
- 训练轮数、densification、medium loss 等命令行参数。

输出：

- `final.ply`：最终 Gaussian checkpoint；
- `medium.pt`：最终 medium field sidecar；
- `best/best_psnr.ply` 和 `best/best_psnr_medium.pt`；
- `best/best_ssim.ply` 和 `best/best_ssim_medium.pt`；
- `checkpoints/step_xxxxxx.ply` 和对应 `step_xxxxxx_medium.pt`；
- `best_metrics.json`；
- `training_summary.json`；
- `train.log`，如果用 `nohup` 重定向。

渲染评估入口：

```bash
methods/ours/render_3dgs_views.py
```

输入：

- `--checkpoint`：`.ply` Gaussian checkpoint；
- `--medium-checkpoint`：对应的 `medium.pt`；
- `--split`：`train` 或 `test`；
- `--lpips`：是否计算 LPIPS。

输出：

- render PNG；
- GT PNG；
- `metrics.csv`。

## 4. 服务器环境

服务器上不需要 `conda activate`，直接使用环境里的 Python：

```bash
/home/user/micromamba/bin/python
```

进入项目：

```bash
cd /root/SJTU
```

当前 Curasao 数据路径：

```bash
/root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole
```

## 5. 推荐稳定训练命令

这是当前最推荐的稳定版训练命令。它跑 12000 轮，但在 6500 步停止 densification，并关闭 opacity reset。

```bash
cd /root/SJTU

/home/user/micromamba/bin/python methods/ours/train_3dgs_scene.py \
  --data /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --out outputs/ours_medium_curasao_12000_dstop6500 \
  --iterations 12000 \
  --eval-every 500 \
  --save-every 1000 \
  --densification-mode standard_3dgs \
  --densify-stop-step 6500 \
  --opacity-reset-interval 0 \
  --lambda-medium 0.001 \
  --lambda-decor 0.01 \
  --medium-samples 16 \
  --grad-clip-norm 1.0 \
  --nonfinite-stop-patience 20
```

为什么这样设：

- `--densify-stop-step 6500`：让 Gaussian clone / split / prune 在 6500 步后停止；
- `--opacity-reset-interval 0`：关闭 opacity reset，降低后期 medium 分支和深度突然变化造成 `nan` 的风险；
- `--eval-every 500`：每 500 步测试一次并自动保存 best checkpoint；
- `--save-every 1000`：保留中间 checkpoint，方便 final 出问题时回退。
- `--grad-clip-norm 1.0`：限制全局梯度范数，降低后期参数突然炸掉的概率；
- `--nonfinite-stop-patience 20`：连续 20 步遇到 `nan/inf` 时提前停止，避免继续浪费时间。

## 6. 后台训练命令

如果想让训练断开终端后继续跑：

```bash
cd /root/SJTU
mkdir -p outputs/ours_medium_curasao_12000_dstop6500

nohup /home/user/micromamba/bin/python methods/ours/train_3dgs_scene.py \
  --data /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --out outputs/ours_medium_curasao_12000_dstop6500 \
  --iterations 12000 \
  --eval-every 500 \
  --save-every 1000 \
  --densification-mode standard_3dgs \
  --densify-stop-step 6500 \
  --opacity-reset-interval 0 \
  --lambda-medium 0.001 \
  --lambda-decor 0.01 \
  --medium-samples 16 \
  --grad-clip-norm 1.0 \
  --nonfinite-stop-patience 20 \
  > outputs/ours_medium_curasao_12000_dstop6500/train.log 2>&1 &
```

查看日志：

```bash
tail -f /root/SJTU/outputs/ours_medium_curasao_12000_dstop6500/train.log
```

查看训练进程：

```bash
pgrep -af "methods/ours/train_3dgs_scene.py"
```

停止训练：

```bash
kill <PID>
```

## 7. 指标测试命令

优先测试 `best` checkpoint，不要只看 `final.ply`。如果训练后段出现 `nan`，`final.ply` 可能不可用，但 `best` 仍然是有效结果。

测试 `best_psnr`：

```bash
cd /root/SJTU

/home/user/micromamba/bin/python methods/ours/render_3dgs_views.py \
  --data /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/ours_medium_curasao_12000_dstop6500/best/best_psnr.ply \
  --medium-checkpoint outputs/ours_medium_curasao_12000_dstop6500/best/best_psnr_medium.pt \
  --out outputs/ours_medium_curasao_12000_dstop6500/test_renders_best_psnr \
  --split test \
  --lpips \
  --lpips-backend official_3dgs
```

测试 `best_ssim`：

```bash
cd /root/SJTU

/home/user/micromamba/bin/python methods/ours/render_3dgs_views.py \
  --data /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/ours_medium_curasao_12000_dstop6500/best/best_ssim.ply \
  --medium-checkpoint outputs/ours_medium_curasao_12000_dstop6500/best/best_ssim_medium.pt \
  --out outputs/ours_medium_curasao_12000_dstop6500/test_renders_best_ssim \
  --split test \
  --lpips \
  --lpips-backend official_3dgs
```

只写指标 CSV，不保存图片：

```bash
/home/user/micromamba/bin/python methods/ours/render_3dgs_views.py \
  --data /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/ours_medium_curasao_12000_dstop6500/best/best_psnr.ply \
  --medium-checkpoint outputs/ours_medium_curasao_12000_dstop6500/best/best_psnr_medium.pt \
  --out outputs/ours_medium_curasao_12000_dstop6500/test_renders_best_psnr_csv_only \
  --split test \
  --lpips \
  --lpips-backend official_3dgs \
  --no-save-images
```

## 8. 参数说明

### medium 参数

- `--disable-medium`：关闭 medium field，退化为普通 3DGS 路径；
- `--medium-lr`：medium MLP 学习率；
- `--medium-samples`：每条 ray 上 medium 积分采样点数；
- `--medium-hidden-dim`：medium MLP hidden dim；
- `--medium-density-bias`：medium extinction density 的 softplus 前置 bias；
- `--medium-far`：无可靠前景深度时的 fallback 积分距离，`0` 表示使用 `scene_extent * 4`；
- `--medium-max-density`：限制 medium attenuation / scattering density 的最大值，默认 `10.0`；
- `--medium-max-optical-depth`：限制 medium 透射率指数的最大光学厚度，默认 `80.0`；
- `--lambda-medium`：medium 平滑和稀疏正则权重；
- `--lambda-beta`：medium density L1 项权重；
- `--lambda-decor`：medium/object 边缘去相关 loss 权重；
- `--medium-warmup-steps`：前 N 步不启用 medium regularization 和 decorrelation。

### densification 参数

- `--densify-start-step`：开始 clone / split / prune 的步数，默认 `500`；
- `--densify-stop-step`：停止 densification 的步数。设为 `0` 时使用默认规则；
- `--densify-interval`：每隔多少步执行一次 densification；
- `--opacity-reset-interval`：每隔多少步重置 opacity，`0` 表示关闭；
- `--disable-densification`：完全关闭 densification。

### 稳定性参数

- `--grad-clip-norm`：全局梯度裁剪阈值，默认 `1.0`，设为 `0` 表示关闭；
- `--nonfinite-stop-patience`：连续多少步出现 `nan/inf` 后提前停止，默认 `20`；
- `--gaussian-min-log-scale`：Gaussian raw log scale 下限，默认 `-12.0`；
- `--gaussian-max-log-scale`：Gaussian raw log scale 上限，默认 `5.0`；
- `--gaussian-max-sh`：SH 系数绝对值上限，默认 `10.0`；
- `--medium-max-param`：medium MLP 参数绝对值上限，默认 `20.0`。

训练循环会自动执行：

- 如果 `render.image`、`depth`、`means2d`、loss 或 gradient 出现 `nan/inf`，跳过当前 `optimizer.step()`；
- 每次 step 后 sanitize Gaussian raw 参数、medium 参数和 Adam 状态；
- densification 的屏幕空间梯度会过滤非有限值，避免 `grad_max=nan` 进入 clone / split / prune 判断。

默认 stop 规则：

$$
\mathrm{densify\_stop}
=
\max(
\min(\mathrm{iterations}-500, 15000),
501
)
$$

例如：

- `--iterations 7000` 且 `--densify-stop-step 0` 时，实际 stop 是 `6500`；
- `--iterations 12000` 且 `--densify-stop-step 0` 时，实际 stop 是 `11500`；
- `--iterations 19999` 且 `--densify-stop-step 0` 时，实际 stop 是 `15000`。

如果想固定在 6500 步停止，必须显式加：

```bash
--densify-stop-step 6500
```

## 9. NaN 处理建议

如果日志中出现：

```text
loss=nan
SSIM=nan
medium=nan
decor=nan
grad_max=nan
```

说明当前模型参数已经出现非有限值。此时：

1. 不要使用 `final.ply` 报指标；
2. 优先使用 `best/best_psnr.ply` 或 `best/best_ssim.ply`；
3. 对应 medium sidecar 必须一起使用；
4. 下一次训练建议关闭 opacity reset，并提前停止 densification。

推荐稳定参数：

```bash
--densify-stop-step 6500 \
--opacity-reset-interval 0
```

## 10. 验证方式

本地语法检查：

```bash
python3 -m py_compile modules/*.py methods/ours/*.py
```

服务器 smoke test：

```bash
cd /root/SJTU

/home/user/micromamba/bin/python methods/ours/train_3dgs_scene.py \
  --data /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --out outputs/ours_medium_smoke \
  --iterations 1 \
  --target-width 320 \
  --eval-every 0 \
  --save-every 1 \
  --medium-samples 4 \
  --medium-chunk-pixels 8192

/home/user/micromamba/bin/python methods/ours/render_3dgs_views.py \
  --data /root/rivermind-data/SJTU/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/ours_medium_smoke/final.ply \
  --medium-checkpoint outputs/ours_medium_smoke/medium.pt \
  --out outputs/ours_medium_smoke/test_renders \
  --split test \
  --target-width 320 \
  --max-images 1 \
  --no-save-images
```

期望：

- 训练能正常保存 `final.ply` 和 `medium.pt`；
- 渲染时输出 `metrics.csv`；
- `pred_image = rgb_object + rgb_medium` 的主路径保持一致；
- `medium_attn`、`medium_bs` 非负；
- `medium_rgb` 在 `[0, 1]`。
