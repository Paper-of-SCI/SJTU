# SeaSplat Render 使用说明

本文档记录本项目里 `methods/seasplat_Render/` 的常用训练、渲染和指标计算指令。原始论文项目说明仍保留在 `README.md`。

## 环境

SeaSplat 的环境在智川云服务器上已经配好了。登录智川云后优先使用下面的激活脚本：

```bash
source /root/rivermind-data/SJTU/methods/seasplat_Render/activate_seasplat_original.sh
```

这个脚本会进入 SeaSplat Render 目录并激活 `.venv`，同时设置 CUDA 11.8 相关环境变量。激活后建议显式设置项目根目录，避免相对路径混乱：

```bash
export SJTU_ROOT=/root/rivermind-data/SJTU
```

本地 WSL 路径可以用：

```bash
cd /home/leo/Projects/SJTU/methods/seasplat_Render
```

注意：推荐在 `methods/seasplat_Render/` 目录下运行 `train.py`、`render_uw.py`、`metrics.py` 和 `convert.py`。

## 数据目录要求

训练目录需要是 COLMAP 格式，至少包含：

```text
DATASET_PATH/
  images/
  sparse/0/
    cameras.bin 或 cameras.txt
    images.bin 或 images.txt
    points3D.bin 或 points3D.txt
```

示例数据路径：

```bash
export DATA=$SJTU_ROOT/src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole
```

如果是原始图片，还没有 COLMAP 结果，才需要用 `convert.py`：

```bash
python convert.py \
  -s "$DATA" \
  --camera OPENCV
```

如果数据目录已经有 `images/` 和 `sparse/0/`，不要再跑 `convert.py`。

## 训练标准 3DGS Baseline

SeaSplat 的 `train.py` 会自动把输出写到：

```text
DATASET_PATH/experiments/<MMDDYYYY>/<exp>/
```

标准 3DGS baseline：

```bash
python train.py \
  -s "$DATA" \
  --exp vanilla_3dgs \
  --iterations 30000 \
  --eval \
  --resolution -1
```

说明：

- `-s` / `--source_path`：数据目录。
- `--exp`：实验名。
- `--eval`：启用 train/test 划分；COLMAP 数据默认每 8 张取 1 张作为 test。
- `--resolution -1`：宽度超过 1600 时自动缩放到 1600，否则保持原图。
- 如果想强制原图训练，用 `--resolution 1`。

## 训练 3D Medium Field 直接 SeaSplat 渲染

从第 10000 步开始启用新的 3D medium field 水下成像模型。该路径直接复用 SeaSplat 原有 `diff_gaussian_rasterization`：

```bash
python train.py \
  -s "$DATA" \
  --exp seasplat_medium \
  --iterations 30000 \
  --eval \
  --resolution -1 \
  --do_seathru \
  --seathru_from_iter 10000
```

这里的 `--do_seathru` 默认不再走旧的 depth-only SeaSplat 网络，而是走：

$$
I(u)
=
\mathrm{rgb\_object}(u)
+
\mathrm{rgb\_medium}(u)
$$

实现采用 `新的渲染公式.md` 第 14 节的最小可行公式：

$$
I(u)
=
\sum_i
T_i^{gs}(u)\alpha_i(u)c_i
\odot
T_m(t_i)
+
I_{\mathrm{back}}(u)
$$

其中直接光项通过“每个 Gaussian 颜色先乘沿相机到 Gaussian 中心积分得到的 \(T_m(t_i)\)”实现，再交给 SeaSplat 原 rasterizer 做 alpha compositing；`rgb_medium` 则沿像素 ray 积分到有效前景深度 \(D_{\mathrm{vis}}\)。当前版本没有引入外部 per-segment visibility-aware backscatter CUDA 合成。

如果要复现实验旧版 SeaSplat depth-only 路径，显式加：

```bash
--legacy_seathru
```

更短的 smoke test：

```bash
python train.py \
  -s "$DATA" \
  --exp seasplat_medium_smoke \
  --iterations 1 \
  --test_iterations 1 \
  --save_iterations 1 \
  --checkpoint_iterations 1 \
  --eval \
  --resolution 4 \
  --do_seathru \
  --seathru_from_iter 0
```

## 从 Checkpoint 续训

训练会保存：

```text
MODEL_PATH/chkpnt<iteration>.pth
```

续训示例：

```bash
export MODEL=$DATA/experiments/<MMDDYYYY>/seasplat

python train.py \
  -s "$DATA" \
  --exp seasplat_resume \
  --iterations 60000 \
  --eval \
  --resolution -1 \
  --do_seathru \
  --seathru_from_iter 10000 \
  --start_checkpoint "$MODEL/chkpnt30000.pth"
```

注意：续训会写到新的 `--exp` 目录，不会覆盖原目录，除非你手动指定同名实验。

## 渲染 Checkpoint

训练完成后先设置模型路径：

```bash
export MODEL=$DATA/experiments/<MMDDYYYY>/seasplat_medium
```

渲染最后一次保存的 iteration：

```bash
python render_uw.py \
  -s "$DATA" \
  -m "$MODEL" \
  --iteration -1 \
  --skip_train \
  --seathru
```

渲染标准 3DGS baseline：

```bash
export MODEL=$DATA/experiments/<MMDDYYYY>/vanilla_3dgs

python render_uw.py \
  -s "$DATA" \
  -m "$MODEL" \
  --iteration -1 \
  --skip_train
```

只渲染 train，不渲染 test：

```bash
python render_uw.py \
  -s "$DATA" \
  -m "$MODEL" \
  --iteration -1 \
  --skip_test \
  --seathru
```

渲染输出：

- `MODEL/test/render/`：无水或 clear object 渲染。
- `MODEL/test/depth/`：深度图。
- `MODEL/test/with_water/`：新公式的最终水下图，只有传 `--seathru` 才有。
- `MODEL/test/rgb_object/`：物体直接光项。
- `MODEL/test/rgb_medium/`：水体散射项。
- `MODEL/test/medium_attn/`：直接光 attenuation 可视化。
- `MODEL/test/medium_bs/`：backscatter density 可视化。
- `MODEL/test/medium_rgb/`：全局 medium scattering color。

旧版 `--legacy_seathru` 渲染仍输出 `no_water/`、`backscatter/`、`attenuation/`。

## 计算指标

SeaSplat 渲染图对 GT 计算 `SSIM`、`PSNR`、`LPIPS`：

```bash
python metrics.py \
  -m "$MODEL" \
  --comparison_dir with_water \
  --skip_train
```

标准 3DGS baseline 用 `render` 目录：

```bash
python metrics.py \
  -m "$MODEL" \
  --comparison_dir render \
  --skip_train
```

指标输出：

- `MODEL/eval_metrics_with_water.json`：`--comparison_dir with_water` 的指标。
- `MODEL/eval_metrics_render.json`：`--comparison_dir render` 的指标。

训练脚本在评估阶段也会写：

```text
MODEL/eval_metrics.json
```

## 常用参数

- `--iterations`：训练总步数，默认 `30000`。
- `--test_iterations`：训练中评估步数列表，默认每 1000 步一次。
- `--save_iterations`：保存 `point_cloud/iteration_*/point_cloud.ply` 的步数列表。
- `--checkpoint_iterations`：保存 `chkpnt*.pth` 的步数列表。
- `--start_checkpoint`：从已有 `chkpnt*.pth` 续训。
- `--do_seathru`：启用新 3D medium field 直接 SeaSplat 水下成像模型。
- `--legacy_seathru`：切回旧 SeaSplat depth-only `BackscatterNet/AttenuateNet` 路径。
- `--seathru_from_iter`：从第几步开始启用水下成像模型。
- `--medium_lr`：medium field 学习率。
- `--medium_samples`：每条 ray 上的 medium 采样数。
- `--medium_hidden_dim`：medium MLP 隐藏层宽度。
- `--medium_density_bias`：medium density softplus 前的 bias。
- `--medium_far`：medium 积分距离；`0` 表示使用 `scene.cameras_extent * 4`。
- `--lambda_medium`：medium sparse/smooth 正则权重。
- `--lambda_beta`：medium density L1 项权重。
- `--lambda_decor`：`rgb_medium` 和 `rgb_object` 边缘 decorrelation 权重。
- `--medium_warmup_steps`：延迟启用 medium 正则的步数。
- `--bs_at_lr`：旧 `--legacy_seathru` 路径的 backscatter / attenuation 网络学习率。
- `--disable_attenuation`：旧 `--legacy_seathru` 路径只保留 backscatter，关闭 attenuation。
- `--use_bs_residual`：旧 `--legacy_seathru` 路径启用 SeaThru 公式里的 residual 项。
- `--resolution -1`：默认策略；宽度超过 1600 自动缩放到 1600。
- `--resolution 1`：保持原图。
- `--resolution 2` / `4` / `8`：按倍率降采样。
- `--images`：指定图像目录名，默认 `images`。
- `--eval`：启用 train/test split。
- `--skip_train` / `--skip_test`：渲染或指标阶段跳过对应 split。

## 推荐实验流程

1. 训练 baseline：

```bash
python train.py -s "$DATA" --exp vanilla_3dgs --iterations 30000 --eval --resolution -1
```

2. 训练 3D medium field 水下模型：

```bash
python train.py -s "$DATA" --exp seasplat_medium --iterations 30000 --eval --resolution -1 --do_seathru --seathru_from_iter 10000
```

3. 渲染新公式 test：

```bash
export MODEL=$DATA/experiments/<MMDDYYYY>/seasplat_medium
python render_uw.py -s "$DATA" -m "$MODEL" --iteration -1 --skip_train --seathru
```

4. 计算新公式指标：

```bash
python metrics.py -m "$MODEL" --comparison_dir with_water --skip_train
```

5. 渲染 baseline test：

```bash
export MODEL=$DATA/experiments/<MMDDYYYY>/vanilla_3dgs
python render_uw.py -s "$DATA" -m "$MODEL" --iteration -1 --skip_train
```

6. 计算 baseline 指标：

```bash
python metrics.py -m "$MODEL" --comparison_dir render --skip_train
```

## 接口边界

- Owner：`methods/seasplat_Render/`。
- Responsibility：提供 3D medium field 直接 SeaSplat 渲染公式的 COLMAP 数据转换、训练、渲染和指标计算入口，并通过 `--legacy_seathru` 保留旧 SeaSplat 路径。
- Input：COLMAP 数据目录、训练 checkpoint、命令行参数。
- Output：`point_cloud/iteration_*/point_cloud.ply`、`chkpnt*.pth`、`medium_*.pt`、渲染 PNG/JPG、指标 JSON。旧 `--legacy_seathru` 额外输出 `backscatter_*.pth`、`attenuate_*.pth`。
- Main flow：`convert.py` 准备 COLMAP 数据，`train.py` 训练 scene Gaussians 和 medium field，`render_uw.py` 渲染 train/test，`metrics.py` 计算指标。
- Dependency direction：入口脚本调用 `scene/`、`gaussian_renderer/`、`deepseecolor/`、`utils/`；3D medium field 渲染只通过 `gaussian_renderer.render_medium(...)` 暴露，入口脚本不直接依赖外部 rasterizer vendor。
- Error semantics：数据目录缺少 `sparse/0` 或 `transforms_train.json` 会无法识别场景；无 CUDA 或 CUDA extension 未编译会在训练或渲染阶段报错；传 `--seathru` 渲染时缺少 `medium_<iteration>.pt` 会报 `FileNotFoundError`；旧 `--legacy_seathru` 缺少 `backscatter_*.pth` 或 `attenuate_*.pth` 会触发断言。
- Impact scope：训练、渲染和文档入口；普通 3DGS baseline 和旧 `--legacy_seathru` 路径保持可用。
- Verification method：在 `sfquant` 环境中运行 `python -m py_compile train.py render_uw.py gaussian_renderer/__init__.py`、`python train.py --help`、`python render_uw.py --help` 检查入口。

## 常见问题

如果渲染时找不到模型，先确认 `MODEL` 指向的是包含 `point_cloud/iteration_*` 的实验目录。

如果 `--seathru` 渲染时报缺少模型文件，确认训练时传过 `--do_seathru`，并且对应 iteration 下存在：

```text
medium_<iteration>.pt
```

如果渲染旧版 `--legacy_seathru` 模型，则需要：

```text
backscatter_<iteration>.pth
attenuate_<iteration>.pth
```

如果想避免自动缩放到宽度 1600，训练和渲染都显式加：

```bash
--resolution 1
```

如果指标脚本找不到 GT，确认实验目录结构是默认的：

```text
DATASET_PATH/experiments/<MMDDYYYY>/<exp>/
```

因为 `metrics.py` 会从实验目录向上找到 `DATASET_PATH/images` 作为 GT。
