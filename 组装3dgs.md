# 组装 3DGS 训练 Curasao

目标：用已经测过的 `utils/` 和 `modules/`，组装一个最小 3D Gaussian Splatting 训练流程，训练数据使用：

```text
src/datasets/SeathruNeRF_dataset/Curasao
```

已提供可运行入口：

```text
methods/3dgs/train_3dgs_curasao.py
```

## 0. 先确认模块测试通过

这一步是为了确认底层积木没坏：数据读取、相机、PLY、GaussianModel、loss、optimizer、densification、renderer 都能按预期工作。

```bash
# 普通测试，renderer 会跳过
jupyter notebook test.ipynb
```

在 notebook 里 `Restart Kernel` 后 `Run All`，最后看到：

```text
✅ 全部必测项通过 / All required tests passed
```

如果要把 CUDA renderer 也测进去：

```bash
RUN_RENDERER_TEST=1 python - <<'PY'
import json
ns = {}
nb = json.load(open("test.ipynb"))
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        exec(compile("".join(cell.get("source", [])), f"test.ipynb:cell{i}", "exec"), ns)
PY
```

## 1. 加载 Curasao 数据

使用：

```python
scene = load_colmap_dataset(
    "src/datasets/SeathruNeRF_dataset/Curasao",
    split="train",
    load_images=False,
    factor=4,
    holdout=8,
    opengl=False,
)
```

这一步完成的功能：

- 读取 `sparse/0/cameras.bin`、`images.bin`、`points3D.bin`。
- 得到训练图像路径、相机位姿 `c2w_matrices`、内参 `fx/fy/cx/cy`。
- 得到 COLMAP 点云 `point_cloud_xyz` 和颜色 `point_cloud_rgb`。
- `factor=4` 会把训练分辨率降到约 `444x295`，先保证能跑起来；全分辨率会更慢、更吃显存。
- `holdout=8` 表示每 8 张留一张，默认训练集是 18 张。
- `opengl=False` 很关键：`gsplat` 这里按 COLMAP/OpenCV 相机坐标投影；如果用 OpenGL 坐标，初始点云会全部投到相机后方，训练会崩或只看到背景。

## 2. 用 COLMAP 点云初始化 GaussianModel

使用：

```python
model = GaussianModel.from_point_cloud(
    scene.point_cloud_xyz,
    scene.point_cloud_rgb,
    sh_degree=3,
    device="cuda",
)
```

这一步完成的功能：

- 每个 COLMAP 3D 点变成一个 Gaussian。
- `means` 来自点云坐标。
- `features_dc` 来自点云 RGB，转成 0 阶 SH 系数。
- `log_scales` 根据近邻距离初始化。
- `quats` 初始化为单位旋转。
- `logit_opacities` 初始化为默认透明度。

## 3. 每次随机取一张训练相机

使用：

```python
camera = Camera.from_scene_data(scene, camera_index, device="cuda", load_image=True)
```

这一步完成的功能：

- 把 `SceneData` 中第 `camera_index` 张图转成 `Camera`。
- 得到 gsplat 需要的 `camera.viewmat` 和 `camera.K`。
- 读取对应 GT 图像，变成 `camera.image`，shape 是 `(3, H, W)`。

## 4. 渲染当前 Gaussian

使用：

```python
render = renderer.render(model, camera)
```

这一步完成的功能：

- 调用 `gsplat.rasterization`。
- 输入 Gaussian 的位置、尺度、旋转、透明度、SH 颜色。
- 输出：
  - `render.image`：当前渲染图，shape `(3, H, W)`。
  - `render.alpha`：透明度图，shape `(1, H, W)`。
  - `render.radii`：每个 Gaussian 的屏幕半径，densification 会用。
  - `render.means2d`：屏幕空间位置，反传后 densification 会读取梯度。

## 5. 计算图像损失

使用：

```python
loss, parts = photometric_loss(render.image.clamp(0, 1), camera.image, lambda_dssim=0.2)
```

这一步完成的功能：

- 计算 3DGS 常用图像损失：

```text
loss = 0.8 * L1 + 0.2 * (1 - SSIM) / 2
```

- `parts["l1"]`、`parts["ssim"]` 用来打印训练状态。

## 6. 反向传播和 Adam 更新

使用：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

这一步完成的功能：

- `loss.backward()` 把图像误差反传到 Gaussian 参数。
- `optimizer.step()` 更新：
  - `means`
  - `features_dc`
  - `features_rest`
  - `logit_opacities`
  - `log_scales`
  - `quats`

脚本里还会给 `means` 设置指数衰减学习率：

```python
set_group_lr(optimizer, "means", position_lr(step))
```

## 7. 自适应致密化 densification

使用：

```python
stats = densifier.update(model, render, optimizer, step)
```

这一步完成的功能：

- 读取 `render.means2d` 的屏幕空间梯度。
- 梯度大、尺度小的 Gaussian 执行 clone。
- 梯度大、尺度大的 Gaussian 执行 split。
- 透明度太低的 Gaussian 执行 prune。
- 同步 optimizer state，避免参数数量变化后 Adam 状态错位。

注意：脚本顺序是先 `optimizer.step()`，再 `densifier.update()`。这样当前 step 的梯度已经用于参数更新，densification 再基于这些梯度调整下一轮的模型结构。

## 8. 保存预览图和 PLY checkpoint

脚本会定期保存：

```text
outputs/curasao_3dgs/previews/step_XXXXXX.png
outputs/curasao_3dgs/checkpoints/step_XXXXXX.ply
outputs/curasao_3dgs/final.ply
```

这一步完成的功能：

- `previews/*.png`：当前训练视角的渲染预览。
- `checkpoints/*.ply`：保存 Gaussian 参数。
- `final.ply`：训练完成后的最终 Gaussian checkpoint。

注意：这里的 `.ply` 不是普通三角网格，也不是传统点云可视化文件；它是 3DGS 的 Gaussian 参数 checkpoint，里面存的是位置、SH 颜色、透明度、尺度、旋转。普通 MeshLab/Blender 不一定能按 3DGS 正确显示。

## 8.1 训练完怎么看结果

最直接看法：用训练好的 PLY 按 Curasao 相机渲染成图片。

```bash
python methods/3dgs/render_3dgs_curasao.py \
  --checkpoint outputs/curasao_3dgs/final.ply \
  --out outputs/curasao_3dgs/renders_test \
  --split test \
  --factor 4
```

输出：

```text
outputs/curasao_3dgs/renders_test/*_render.png
outputs/curasao_3dgs/renders_test/*_gt.png
```

你打开 `*_render.png` 看模型渲染结果，打开 `*_gt.png` 看真实图对比。终端也会打印每张图 PSNR 和平均 PSNR。

如果只想快速看一张：

```bash
python methods/3dgs/render_3dgs_curasao.py \
  --checkpoint outputs/curasao_3dgs/final.ply \
  --out outputs/curasao_3dgs/renders_one \
  --split test \
  --factor 4 \
  --max-images 1
```

## 9. 先跑一个 smoke test

先用很少迭代确认训练链路能跑通：

```bash
python methods/3dgs/train_3dgs_curasao.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao \
  --out outputs/curasao_smoke \
  --iterations 5 \
  --factor 8 \
  --save-every 5 \
  --log-every 1 \
  --disable-densification
```

这一步只验证：

- 数据能加载。
- 相机能构造。
- renderer 能渲染。
- loss 能反传。
- optimizer 能更新。
- checkpoint 能保存。

## 10. 正式训练 Curasao

确认 smoke test 没问题后，跑正式训练：

```bash
python methods/3dgs/train_3dgs_curasao.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao \
  --out outputs/curasao_3dgs \
  --iterations 7000 \
  --factor 4 \
  --save-every 1000 \
  --log-every 50
```

如果显存不够，把 `factor` 调大：

```bash
--factor 8
```

如果你只想先看普通优化，不想让 Gaussian 数量变化：

```bash
--disable-densification
```

## 11. 看训练是否正常

终端会用中文输出，并用 `tqdm` 显示进度条、速度和预计剩余时间，类似：

```text
设备：CUDA GPU='NVIDIA GeForce RTX 3070 Laptop GPU' 显存=8.00GB
数据划分：训练=18 张，测试=3 张，holdout=8，分辨率=444x295，降采样 factor=4
训练进度:  10%|█         | 700/7000 [03:20<30:10, 3.48步/s, loss=..., 训练PSNR=..., G=...]
第 000700 步 | loss=... | L1=... | SSIM=... | 训练PSNR=... | Gaussian=...
测试评估 | 第 001000 步 | 测试图像=3 | 测试PSNR=...
```

你主要看：

- `设备：CUDA ...`：说明确实在用 GPU，并显示 GPU 型号和显存。
- `数据划分：训练=18 张，测试=3 张`：说明已经按 `holdout=8` 分了训练集和测试集。
- `tqdm` 进度条右侧会显示速度和预计剩余时间。
- `loss` 是否总体下降。
- `train_psnr` 是否总体上升。
- `test_psnr` 是测试集 PSNR，用来看泛化效果；由 `--eval-every` 控制频率。
- `gaussians` 是否在 densification 后增加或被 prune。
- `outputs/.../previews` 里的图是否逐渐接近 GT。

## 13. GPU 利用率为什么可能只有 30% 多

这不一定代表没用 GPU。脚本启动时会打印 `设备：CUDA GPU=...`，这说明渲染和反传在 GPU 上。

GPU 利用率低常见原因：

- `factor` 太大，图像分辨率低，例如 `factor=16` 只有 `111x73`，GPU 工作量太小。
- 当前是单视角逐步训练，每次只渲染一张图，不是 batch 多图并行。
- Gaussian 数量初始约 2.6 万，还不算特别大。
- 每隔 `--eval-every` 做测试集评估，会插入额外同步。
- 保存 preview/PLY 时会从 GPU 拷贝到 CPU，也会让 GPU 等待。

想提高 GPU 利用率：

- 正式训练用 `--factor 4` 或更小，不要长期用 `--factor 16`。
- 把 `--eval-every` 调大，例如 `--eval-every 1000`。
- 把 `--save-every` 调大，例如 `--save-every 1000`。
- densification 开启后 Gaussian 数量增加，GPU 利用率通常会上升。

## 12. 训练脚本的整体流程

完整训练循环就是：

```text
加载 Curasao
  -> 点云初始化 GaussianModel
  -> 随机取一张 Camera
  -> renderer.render()
  -> photometric_loss()
  -> loss.backward()
  -> optimizer.step()
  -> densifier.update()
  -> 保存 preview / PLY
```

每个模块只负责自己的职责：

- `utils.dataset_loaders`：只负责读数据。
- `modules.camera`：只负责相机矩阵和 GT 图像。
- `modules.gaussian_model`：只负责 Gaussian 参数。
- `modules.renderer`：只负责渲染。
- `modules.losses`：只负责损失。
- `modules.optim`：只负责 optimizer 和 LR。
- `modules.densification`：只负责 clone/split/prune。
- `utils.ply_io`：只负责保存 Gaussian checkpoint。
