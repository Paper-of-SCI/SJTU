# SeaSplat_Densify 使用说明

本文档记录本项目里 `methods/seasplat_Densify/` 的常用训练、渲染和指标计算指令。原始论文项目说明仍保留在 `README.md`。

## 环境

SeaSplat 的环境在智川云服务器上已经配好了。登录智川云后优先使用下面的激活脚本：

```bash
source /root/rivermind-data/SJTU/methods/seasplat_Densify/activate_seasplat_original.sh
```

这个脚本会进入 SeaSplat 目录并激活 `.venv`，同时设置 CUDA 11.8 相关环境变量。激活后建议显式设置项目根目录，避免相对路径混乱：

```bash
export SJTU_ROOT=/root/rivermind-data/SJTU
```

本地 WSL 路径可以用：

```bash
cd /home/leo/Projects/SJTU/methods/seasplat_Densify
```

注意：SeaSplat_Densify 保留 SeaSplat 项目结构，推荐在 `methods/seasplat_Densify/` 目录下运行 `train.py`、`render_uw.py`、`metrics.py` 和 `convert.py`。

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

## 训练 SeaSplat

从第 10000 步开始启用 SeaThru / SeaSplat 水下成像模型：

```bash
python train.py \
  -s "$DATA" \
  --exp seasplat \
  --iterations 30000 \
  --eval \
  --resolution -1 \
  --do_seathru \
  --seathru_from_iter 10000
```

更短的 smoke test：

```bash
python train.py \
  -s "$DATA" \
  --exp seasplat_smoke \
  --iterations 1 \
  --test_iterations 1 \
  --save_iterations 1 \
  --checkpoint_iterations 1 \
  --eval \
  --resolution 4 \
  --do_seathru \
  --seathru_from_iter 1
```

## 训练 Patch-Guided Densify

`methods/seasplat_Densify/` 默认启用 patch-guided densification，只替换 clone / split / prune 的分数来源，SeaSplat 的渲染、loss、SeaThru 网络和输出目录结构保持不变。

```bash
python train.py \
  -s "$DATA" \
  --exp seasplat_patch_densify \
  --iterations 30000 \
  --eval \
  --resolution -1 \
  --densification_mode patch_guided \
  --patch_size 32 \
  --patch_edge_weight 0.75 \
  --patch_detail_lambda 2.0 \
  --clone_jitter_scale 0.05
```

标准 3DGS densification 对照：

```bash
python train.py \
  -s "$DATA" \
  --exp seasplat_standard_densify \
  --iterations 30000 \
  --eval \
  --resolution -1 \
  --densification_mode standard_3dgs
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
export MODEL=$DATA/experiments/<MMDDYYYY>/seasplat
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

- `MODEL/test/render/`：普通 3DGS RGB 渲染。
- `MODEL/test/depth/`：深度图。
- `MODEL/test/with_water/`：SeaSplat 合成的水下图，只有传 `--seathru` 才有。
- `MODEL/test/no_water/`：去水效果图，只有传 `--seathru` 才有。
- `MODEL/test/backscatter/`：backscatter 可视化，只有传 `--seathru` 才有。
- `MODEL/test/attenuation/`：attenuation 可视化，只有传 `--seathru` 才有。

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
- `--densification_mode`：densification 策略，`patch_guided` 为默认主路径，`standard_3dgs` 保留原 SeaSplat / 3DGS 对照路径。
- `--patch_size`：patch detail 的池化窗口，默认 `32`。
- `--patch_edge_weight`：GT 边缘对 patch detail 的加权，默认 `0.75`。
- `--patch_detail_lambda`：patch detail 对屏幕空间梯度的调制强度，默认 `2.0`。
- `--clone_jitter_scale`：patch-guided clone 的局部扰动强度，默认 `0.05`。
- `--test_iterations`：训练中评估步数列表，默认每 1000 步一次。
- `--save_iterations`：保存 `point_cloud/iteration_*/point_cloud.ply` 的步数列表。
- `--checkpoint_iterations`：保存 `chkpnt*.pth` 的步数列表。
- `--start_checkpoint`：从已有 `chkpnt*.pth` 续训。
- `--do_seathru`：启用 SeaSplat 水下成像模型。
- `--seathru_from_iter`：从第几步开始启用 SeaSplat。
- `--bs_at_lr`：backscatter / attenuation 网络学习率。
- `--disable_attenuation`：只保留 backscatter，关闭 attenuation。
- `--use_bs_residual`：启用 SeaThru 公式里的 residual 项。
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

2. 训练 SeaSplat：

```bash
python train.py -s "$DATA" --exp seasplat --iterations 30000 --eval --resolution -1 --do_seathru --seathru_from_iter 10000
```

3. 渲染 SeaSplat test：

```bash
export MODEL=$DATA/experiments/<MMDDYYYY>/seasplat
python render_uw.py -s "$DATA" -m "$MODEL" --iteration -1 --skip_train --seathru
```

## 接口边界

- Owner：`methods/seasplat_Densify/`。
- Responsibility：基于 SeaSplat 训练入口提供 patch-guided densification 变体。
- Input：COLMAP 场景目录、SeaSplat CLI 参数和 densification 参数。
- Output：SeaSplat 原有 checkpoint、PLY、渲染图和指标文件。
- Main flow：`train.py` 训练时选择 `standard_3dgs` 或 `patch_guided` densification；`render_uw.py` 和 `metrics.py` 不变。
- Dependency direction：训练入口调用 `scene.densification` 和 `scene.gaussian_model`；densification 模块不依赖训练 loss、渲染保存或评估逻辑。
- Error semantics：非法 `--densification_mode` 会抛 `ValueError`；patch detail 输入图像 shape 不匹配会抛 `ValueError`。
- Impact scope：只影响 `methods/seasplat_Densify/`，不修改 `methods/seasplat/`。
- Verification method：运行 `python train.py --help`、`python -m py_compile train.py arguments/__init__.py scene/gaussian_model.py scene/densification.py`，再跑 1 step smoke test。

4. 计算 SeaSplat 指标：

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

- Owner：`methods/seasplat/`。
- Responsibility：提供原版 SeaSplat 的 COLMAP 数据转换、训练、渲染和指标计算入口。
- Input：COLMAP 数据目录、训练 checkpoint、命令行参数。
- Output：`point_cloud/iteration_*/point_cloud.ply`、`chkpnt*.pth`、`backscatter_*.pth`、`attenuate_*.pth`、渲染 PNG/JPG、指标 JSON。
- Main flow：`convert.py` 准备 COLMAP 数据，`train.py` 训练模型，`render_uw.py` 渲染 train/test，`metrics.py` 计算指标。
- Dependency direction：入口脚本调用 `scene/`、`gaussian_renderer/`、`deepseecolor/`、`utils/`，下层模块不依赖入口脚本。
- Error semantics：数据目录缺少 `sparse/0` 或 `transforms_train.json` 会无法识别场景；无 CUDA 或 CUDA extension 未编译会在训练或渲染阶段报错；传 `--seathru` 渲染时缺少 `backscatter_*.pth` 或 `attenuate_*.pth` 会触发断言。
- Impact scope：本文档只新增使用说明，不修改 SeaSplat 代码路径。
- Verification method：在智川云环境中运行 `python train.py --help`、`python render_uw.py --help`、`python metrics.py --help` 检查命令行入口。

## 常见问题

如果渲染时找不到模型，先确认 `MODEL` 指向的是包含 `point_cloud/iteration_*` 的实验目录。

如果 `--seathru` 渲染时报缺少模型文件，确认训练时传过 `--do_seathru`，并且对应 iteration 下存在：

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
