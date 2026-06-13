# 3DGS 用法

本文档记录 `methods/ours_denstify/` 下面现有 3DGS 训练、渲染和 benchmark 入口的常用指令。
本目录只负责编排训练、渲染和 benchmark；densification 算法统一由 `modules/densification.py` 拥有。

## 基本约定

从项目根目录运行：

```bash
cd /home/leo/Projects/SJTU
conda activate sfquant
```

下面命令默认使用这个数据目录：

```bash
src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole
```

如果换场景，只需要替换 `--data` 或 benchmark 里的 `--scenes`。

## 训练单个场景

标准 3DGS 训练：

```bash
python methods/ours_denstify/train_3dgs_scene.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --out outputs/3dgs_scene/Curasao_undistorted \
  --iterations 7000 \
  --eval-every 500 \
  --save-every 1000 \
  --densification-mode standard_3dgs
```

训练输出：

- `outputs/3dgs_scene/Curasao_undistorted/final.ply`：最终 3DGS 模型。
- `outputs/3dgs_scene/Curasao_undistorted/checkpoints/step_*.ply`：按 `--save-every` 保存的中间 checkpoint。
- `outputs/3dgs_scene/Curasao_undistorted/previews/step_*.png`：训练过程预览图。
- `outputs/3dgs_scene/Curasao_undistorted/best/best_psnr.ply`：按 held-out PSNR 记录的最好 checkpoint。
- `outputs/3dgs_scene/Curasao_undistorted/best_metrics.json`：训练中 best 指标记录。
- `outputs/3dgs_scene/Curasao_undistorted/training_summary.json`：训练配置和结果摘要。

快速检查训练流程：

```bash
python methods/ours_denstify/train_3dgs_scene.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --out outputs/3dgs_scene/Curasao_smoke \
  --iterations 1 \
  --eval-every 0 \
  --save-every 1
```

## 测试集渲染和指标

渲染 final checkpoint：

```bash
python methods/ours_denstify/render_3dgs_views.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/3dgs_scene/Curasao_undistorted/final.ply \
  --out outputs/3dgs_scene/Curasao_undistorted/test_renders_final \
  --split test \
  --lpips \
  --lpips-backend official_3dgs
```

渲染 best PSNR checkpoint：

```bash
python methods/ours_denstify/render_3dgs_views.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/3dgs_scene/Curasao_undistorted/best/best_psnr.ply \
  --out outputs/3dgs_scene/Curasao_undistorted/test_renders_best_psnr \
  --split test \
  --lpips \
  --lpips-backend official_3dgs
```

只快速测一张图，不保存 PNG：

```bash
python methods/ours_denstify/render_3dgs_views.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/3dgs_scene/Curasao_undistorted/final.ply \
  --out outputs/3dgs_scene/Curasao_undistorted/test_renders_smoke \
  --split test \
  --max-images 1 \
  --no-save-images
```

测试输出：

- `render_*.png`：渲染图。
- `gt_*.png`：对应 GT 图。
- `metrics.csv`：逐图和平均 `PSNR`、`SSIM`、`L1`，开启 `--lpips` 后还包含 `LPIPS`。

## 新视角路径渲染

在两个训练相机之间插值：

```bash
python methods/ours_denstify/render_3dgs_path.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/3dgs_scene/Curasao_undistorted/final.ply \
  --out outputs/3dgs_scene/Curasao_undistorted/path_interpolate \
  --mode interpolate \
  --split train \
  --frames 60 \
  --start 0 \
  --end -1
```

绕场景 orbit 渲染：

```bash
python methods/ours_denstify/render_3dgs_path.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
  --checkpoint outputs/3dgs_scene/Curasao_undistorted/final.ply \
  --out outputs/3dgs_scene/Curasao_undistorted/path_orbit \
  --mode orbit \
  --split train \
  --frames 120 \
  --radius-scale 1.0
```

路径渲染输出为 `frame_0000.png`、`frame_0001.png` 等逐帧图片。

## Patch Densification Benchmark

对比标准 3DGS、patch-guided 和 semantic patch-guided densification：

```bash
python methods/ours_denstify/benchmark_patch_densification.py \
  --data-root src/datasets/SeathruNeRF_dataset \
  --scenes Curasao/undistorted_pinhole \
  --out outputs/3dgs_patch_curasao \
  --variants standard_3dgs patch_guided patch_guided_semantic \
  --seeds 0 \
  --iterations 7000 \
  --train-eval-every 1000 \
  --lpips \
  --lpips-backend official_3dgs \
  --patch-size 32 \
  --semantic-importance-root semantic_importance \
  --compact-summary
```

只要 CSV 指标、不保留渲染 PNG 和最终 PLY：

```bash
python methods/ours_denstify/benchmark_patch_densification.py \
  --data-root src/datasets/SeathruNeRF_dataset \
  --scenes Curasao/undistorted_pinhole \
  --out outputs/3dgs_patch_curasao_csv \
  --variants standard_3dgs patch_guided patch_guided_semantic \
  --seeds 0 \
  --iterations 7000 \
  --train-eval-every 1000 \
  --patch-size 32 \
  --semantic-importance-root semantic_importance \
  --csv-only \
  --compact-summary
```

reallocate 只建议作为诊断实验，并显式降低重分配比例：

```bash
python methods/ours_denstify/benchmark_patch_densification.py \
  --data-root src/datasets/SeathruNeRF_dataset \
  --scenes Curasao/undistorted_pinhole \
  --out outputs/3dgs_patch_curasao_reallocate_diag \
  --variants standard_3dgs patch_reallocate patch_reallocate_semantic \
  --seeds 0 \
  --iterations 7000 \
  --train-eval-every 1000 \
  --patch-size 32 \
  --semantic-importance-root semantic_importance \
  --reallocate-fraction 0.005 \
  --compact-summary
```

可选 `--variants`：

- `standard`：`standard_3dgs` 的别名。
- `standard_3dgs`：标准 clone、split、prune densification。
- `patch_guided`：用 patch 细节分数调整 densification。
- `patch_reallocate`：在 patch-guided 基础上重分配低细节 Gaussian。
- `patch_guided_semantic`：需要 `--semantic-importance-root`。
- `patch_reallocate_semantic`：需要 `--semantic-importance-root`。

benchmark 输出：

- `summary.csv`：每个 scene、variant、seed 的测试指标。
- `summary.json`：同一份结果的 JSON 版本。
- `summary_agg.csv`：开启 `--compact-summary` 后生成的聚合均值和标准差。
- 每次运行的训练结果在 `outputs/3dgs_patch_curasao/<scene>/<variant>/seed_<seed>/train/`。
- 每次运行的测试结果在 `outputs/3dgs_patch_curasao/<scene>/<variant>/seed_<seed>/test_renders/`。

## 常用参数

- `--target-width` / `--target-height`：指定渲染或训练分辨率，优先级高于 `--factor`。
- `--factor`：图像降采样倍率；默认 `-1`，宽度超过 1600 时会自动缩放。
- `--holdout`：每隔多少张图划到测试集，默认 `8`。
- `--holdout-offset`：held-out 采样偏移，默认 `0`。
- `--eval-every`：训练时每隔多少步评估测试集；设为 `0` 可关闭 best checkpoint 跟踪。
- `--eval-lpips`：训练评估时计算 LPIPS。
- `--lpips`：测试渲染时计算 LPIPS。
- `--lpips-backend official_3dgs`：使用和官方 3DGS `metrics.py` 对齐的 LPIPS 实现。
- `--lpips-backend seasplat`：使用 `methods/seasplat/lpipsPyTorch` 的 LPIPS 实现；benchmark 中所有 variant 会使用同一个 backend。
- `--disable-densification`：关闭 clone、split、prune。
- `--opacity-reset-interval 0`：关闭 opacity reset。
- `--densify-start-step`、`--densify-stop-step`、`--densify-interval`：控制 densification 的时间窗口和频率。

## 接口边界

- Owner：`methods/ours_denstify/`。
- Responsibility：提供 3DGS 训练、checkpoint 渲染、新视角路径渲染和 patch densification 对比实验入口；densification 算法由 `modules/densification.py` 统一实现。
- Input：COLMAP 场景目录、PLY checkpoint、命令行参数。
- Output：PLY checkpoint、PNG 渲染图、`metrics.csv`、`summary.csv/json`、`training_summary.json`。
- Main flow：`train_3dgs_scene.py` 训练模型，`render_3dgs_views.py` 评估 checkpoint，`render_3dgs_path.py` 渲染新视角，`benchmark_patch_densification.py` 串联训练和测试。
- Dependency direction：入口脚本调用 `modules/` 和 `utils/`，下层模块不依赖 `methods/ours_denstify/`。
- Error semantics：路径不存在会抛 `FileNotFoundError`；无 CUDA 会抛 `RuntimeError`；semantic variant 缺少 `--semantic-importance-root` 会抛 `ValueError`。
- Impact scope：本文档只描述用法，不改变训练、渲染或 benchmark 行为。
- Verification method：运行 `python methods/ours_denstify/train_3dgs_scene.py --help`、`python methods/ours_denstify/render_3dgs_views.py --help`、`python methods/ours_denstify/benchmark_patch_densification.py --help` 检查入口参数。

### Peer Module Call Exception

- Trigger:
  - `--lpips-backend seasplat` 需要复用 `methods/seasplat/lpipsPyTorch` 的 vendored LPIPS 实现，保证和 SeaSplat 指标代码调用同一套 LPIPS。
- Dependency direction:
  - `methods/ours_denstify/lpips_backend.py` -> `methods/seasplat/lpipsPyTorch`。
  - `methods/seasplat` 不依赖 `methods/ours_denstify`。
- Contract:
  - Input：`image` 和 `gt` 为 CHW tensor，值域 `[0, 1]`；`net` 为 `alex`、`vgg` 或 `squeeze`。
  - Output：单个 float LPIPS 值。
  - Error semantics：SeaSplat LPIPS 导入失败时抛 `ImportError`；非法 `net` 抛 `ValueError`。
  - Boundary conditions：只在显式选择 `--lpips-backend seasplat` 时启用，不影响默认 LPIPS backend。
- Impact scope:
  - Modules affected：`methods/ours_denstify` 训练、渲染、benchmark 入口。
  - Modules not affected：`modules/densification.py`、renderer、loss、dataset loader。
- Exit plan:
  - 如果后续把 SeaSplat LPIPS 提升为仓库级共享 metric adapter，可移除这个 peer 调用，让所有方法统一依赖共享 metric entry。

## 常见问题

如果提示找不到路径，先确认命令是在 `/home/leo/Projects/SJTU` 下运行，并且 `--data` 指向真实存在的 COLMAP 场景目录。

如果 CUDA 报错或显存不够，先降低分辨率，例如加：

```bash
--target-width 800
```

如果只想看测试指标，不想保存大量 PNG，在 `render_3dgs_views.py` 或 benchmark 的 `--csv-only` 模式里使用 `--no-save-images` 或 `--csv-only`。
