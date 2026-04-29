# modules/

基于 PyTorch 的 3D Gaussian Splatting 功能模块。

**每个文件负责单一职责，依赖关系无环，可按需独立替换。**  
外部仅需安装 `diff-gaussian-rasterization` CUDA 后端（仅 `renderer.py` 直接调用）。

---

## 依赖安装

```bash
# diff-gaussian-rasterization（官方 3DGS CUDA 后端）
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization

# 其他标准依赖
pip install torch torchvision
```

---

## 模块一览

| 文件 | 功能 | 主要类 / 函数 |
|------|------|--------------|
| `spherical_harmonics.py` | 球谐函数（0~3 阶） | `eval_sh`, `rgb_to_sh_dc`, `sh_dc_to_rgb`, `get_view_directions` |
| `losses.py` | 损失函数 | `l1_loss`, `ssim_loss`, `photometric_loss`, `depth_regularization_loss`, `opacity_entropy_loss` |
| `camera.py` | 相机数据类 | `Camera`, `camera_from_scene_data`, `cameras_from_scene_data` |
| `gaussian_model.py` | 高斯场景模型（核心） | `GaussianModel` |
| `renderer.py` | 可微分光栅化器 | `GaussianRenderer`, `RenderOutput` |
| `densification.py` | 自适应密度控制 | `DensificationController`, `DensificationStats` |
| `training.py` | 训练主循环 | `TrainingConfig`, `build_optimizer`, `training_step`, `evaluate`, `train` |

---

## 依赖关系图

```
utils/camera_utils ──┐
utils/dataset_loaders┤──→ camera.py
                     │
utils/ply_io ────────┼──→ gaussian_model.py
                     │
spherical_harmonics ─┤
losses ──────────────┤──→ (被 training.py 使用)
camera ──────────────┤
gaussian_model ──────┼──→ renderer.py
                     │
renderer ────────────┤
gaussian_model ──────┼──→ densification.py
                     │
所有 modules ────────┘──→ training.py
```

---

## 各模块详细说明

### `spherical_harmonics.py` — 球谐函数

解析式实球谐（degree 0~3），与官方 3DGS 仓库系数完全一致。

```python
from modules.spherical_harmonics import eval_sh, rgb_to_sh_dc, get_view_directions

# 视角相关颜色计算
view_dirs = get_view_directions(means, camera_center)   # (N, 3) 单位向量
colors = eval_sh(degree=3, sh_coeffs=coeffs, directions=view_dirs)  # (N, 3)
colors = (colors + 0.5).clamp(0)    # 加 0.5 偏移后 clamp

# RGB ↔ SH DC 系数
sh_dc = rgb_to_sh_dc(rgb_tensor)    # (N, 3)  初始化时使用
rgb   = sh_dc_to_rgb(sh_dc)         # 反向
```

**SH 系数规格：**
- `sh_coeffs` 形状：`(N, (degree+1)^2, 3)`
- 支持 0、1、2、3 阶，分别对应 1、4、9、16 个系数

---

### `losses.py` — 损失函数

```python
from modules.losses import photometric_loss, depth_regularization_loss

# 标准 3DGS 光度损失（论文默认 λ=0.2）
total, comps = photometric_loss(pred, gt, lambda_dssim=0.2)
# comps: {'l1': ..., 'ssim': ..., 'dssim': ..., 'total': ...}

# 水下场景深度平滑正则（可选）
reg = depth_regularization_loss(depth_map)
```

所有函数均为可微分 PyTorch 运算，无副作用。

---

### `camera.py` — 相机数据类

```python
from modules.camera import Camera, cameras_from_scene_data
from utils.dataset_loaders import load_colmap_dataset

scene = load_colmap_dataset("src/datasets/SeathruNeRF_dataset/Panama")
cameras = cameras_from_scene_data(scene, device="cuda")

cam = cameras[0]
print(cam.fov_x, cam.fov_y)            # 水平/垂直视场角（弧度）
print(cam.projection_matrix.shape)     # (4, 4)
print(cam.camera_center.shape)         # (3,)
print(cam.image.shape)                 # (3, H, W)，若 load_images=True
```

`Camera` 是冻结数据类（`frozen=True`），投影矩阵等属性按需计算，不存储。

---

### `gaussian_model.py` — 高斯场景模型（核心）

```python
from modules.gaussian_model import GaussianModel

# 从 COLMAP 稀疏点云初始化
gaussians = GaussianModel.from_point_cloud(xyz, rgb, max_sh_degree=3, device="cuda")
print(gaussians.num_gaussians)         # 初始高斯数量

# 属性访问（均带激活函数）
gaussians.means        # (N, 3)，无激活
gaussians.scales       # (N, 3)，exp 激活
gaussians.rotations    # (N, 4)，L2 归一化
gaussians.opacities    # (N, 1)，sigmoid 激活
gaussians.sh_coefficients  # (N, (degree+1)^2, 3)

# 协方差计算
cov3d = gaussians.compute_covariance_3d()  # (N, 6) 上三角

# 存读 PLY checkpoint
gaussians.save_ply("output/point_cloud.ply")
gaussians = GaussianModel.load_ply("output/point_cloud.ply", device="cuda")
```

**内参数（均为 `nn.Parameter`）：**
| 参数 | 形状 | 激活 |
|------|------|------|
| `_means` | (N, 3) | 无 |
| `_scales` | (N, 3) | `exp` |
| `_rotations` | (N, 4) | L2 norm |
| `_opacities` | (N, 1) | `sigmoid` |
| `_sh_dc` | (N, 1, 3) | 无 |
| `_sh_rest` | (N, K, 3) | 无 |

---

### `renderer.py` — 可微分光栅化器

```python
from modules.renderer import GaussianRenderer

renderer = GaussianRenderer(sh_degree=3, bg_color=torch.ones(3))
out = renderer.render(gaussians, camera)

out.image               # (3, H, W) 渲染 RGB
out.radii               # (N,) 2D 投影半径
out.visibility_filter   # (N,) bool 可见性掩码
out.screenspace_means   # (N, 2)，保留梯度供致密化使用
```

内部流程：`eval_sh` → view-dependent 颜色 → `compute_covariance_3d` → 调用 `GaussianRasterizer`。

---

### `densification.py` — 自适应密度控制

```python
from modules.densification import DensificationController

ctrl = DensificationController(
    gaussians,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_grad_threshold=0.0002,
    scene_extent=scene.scene_scale,
)

# 在训练循环中每步调用
stats = ctrl.update(step, render_output, camera, optimizer)
print(f"克隆 {stats.num_cloned}，分裂 {stats.num_split}，剪除 {stats.num_pruned}")
```

**关键细节：** 参数变异（clone/split/prune）后，`DensificationController` 会自动修补 Adam 优化器的 `exp_avg`、`exp_avg_sq` 动量缓存，保证维度对齐。

---

### `training.py` — 训练主循环

```python
from modules.training import TrainingConfig, train

# 最简单的端到端训练
config = TrainingConfig(
    num_iterations=30000,
    sh_degree=3,
    output_dir="output/experiment_1",
    device="cuda",
)
gaussians = train(
    data_dir="src/datasets/SeathruNeRF_dataset/Curasao",
    config=config,
    scene_format="colmap",   # 或 "llff" / "blender"
)
```

也可单步使用：

```python
from modules.training import build_optimizer, training_step, evaluate, get_expon_lr_func

optimizer = build_optimizer(gaussians, config)
lr_func   = get_expon_lr_func(lr_init=1.6e-4, lr_final=1.6e-6, max_steps=30000)

for step in range(1, 30001):
    camera = random.choice(train_cameras)
    metrics = training_step(gaussians, renderer, camera, optimizer, ctrl, config, step, lr_func)

eval_metrics = evaluate(gaussians, renderer, test_cameras, output_dir="output/eval")
print(f"PSNR={eval_metrics['psnr']:.2f} dB")
```

**`TrainingConfig` 关键超参：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `position_lr_init` | 1.6e-4 | 位置初始学习率 |
| `position_lr_final` | 1.6e-6 | 位置最终学习率 |
| `feature_lr` | 0.0025 | SH DC 学习率（高阶 ÷ 20）|
| `num_iterations` | 30000 | 总训练步数 |
| `sh_degree` | 3 | 最大 SH 阶数 |
| `densify_grad_threshold` | 0.0002 | 2D 梯度致密化阈值 |
| `lambda_dssim` | 0.2 | DSSIM 损失权重 |

---

## 快速开始（端到端）

```python
from modules.training import TrainingConfig, train

gaussians = train(
    data_dir="src/datasets/SeathruNeRF_dataset/Curasao",
    config=TrainingConfig(
        num_iterations=30000,
        output_dir="output/curasao",
        device="cuda",
    ),
    scene_format="colmap",
)
```

训练完成后，checkpoint 保存在：
```
output/curasao/point_cloud/iteration_30000/point_cloud.ply
```

加载并继续渲染：
```python
from modules.gaussian_model import GaussianModel
from modules.renderer import GaussianRenderer

gaussians = GaussianModel.load_ply("output/curasao/point_cloud/iteration_30000/point_cloud.ply")
renderer  = GaussianRenderer(sh_degree=3)
out = renderer.render(gaussians, camera)
```
