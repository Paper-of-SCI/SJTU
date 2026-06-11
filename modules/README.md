# modules/

这里是可组装的 gsplat 版 3DGS 功能模块，不包含固定训练入口。

## 模块职责

| 模块 | 主要职责 | 核心接口 | 依赖关系 |
|---|---|---|---|
| `gaussian_model.py` | 管理 3D Gaussian 的可学习参数；从点云初始化；提供 clone、split、prune 等基础变异操作。 | `GaussianModel`, `GaussianModel.from_point_cloud`, `activated_tensors` | 依赖 `torch`、`modules.spherical_harmonics` |
| `renderer.py` | 对 `gsplat.rasterization` 做一层薄封装；输入模型和相机，输出 RGB、alpha、depth、radii、means2d、metadata。 | `GaussianRenderer`, `RenderOutput`, `GaussianRenderer.render` | 依赖 `torch`、`gsplat`、`GaussianModel`、`Camera` |
| `medium_field.py` | 管理低容量 3D 水体介质场；输入 3D 点，输出 extinction density、backscatter density 和全局 medium 颜色。 | `MediumField`, `MediumFieldConfig`, `scene_medium_config` | 依赖 `torch` |
| `medium_renderer.py` | 在普通 3DGS 渲染结果上执行可微 medium 后处理；输出 `rgb_object`、`rgb_medium`、`pred_image` 和 medium 图。 | `MediumRenderer`, `MediumRenderConfig`, `MediumRenderOutput` | 依赖 `GaussianRenderer`、`MediumField`、`Camera` |
| `camera.py` | 表示单个针孔相机；保存内参、外参、分辨率和可选 GT 图像；生成 gsplat 需要的 `viewmat` 和 `K`。 | `Camera`, `Camera.from_scene_data`, `viewmat`, `K` | 依赖 `torch`；可从 `utils.dataset_loaders.SceneData` 构造 |
| `losses.py` | 提供无状态、可微分的训练损失；不绑定任何训练循环。 | `l1_loss`, `ssim`, `photometric_loss`, `underwater_loss`, `medium_decorrelation_loss` | 依赖 `torch` |
| `densification.py` | 提供外部训练循环可调用的自适应致密化逻辑；根据屏幕空间梯度执行 clone、split、prune，并同步 optimizer state。 | `DensificationConfig`, `DensificationController.update`, `DensificationStats` | 依赖 `torch`、`GaussianModel`、`RenderOutput` |
| `optim.py` | 提供可选的 3DGS Adam 参数组构造和学习率调度工具；外部训练循环可选择使用或替换。 | `OptimConfig`, `build_3dgs_optimizer`, `exponential_lr`, `set_group_lr` | 依赖 `torch`、`GaussianModel` |
| `spherical_harmonics.py` | 提供 RGB 与 0 阶 SH 系数转换，以及 SH basis 数量计算；渲染时 SH 求值交给 gsplat。 | `num_sh_bases`, `rgb_to_sh`, `sh_to_rgb` | 依赖 `torch` |

## 单模块测试方式

| 模块 | 测试命令 | 预期结果 | 备注 |
|---|---|---|---|
| 全部语法 | `python - <<'PY'`<br>`from pathlib import Path`<br>`for p in list(Path('modules').glob('*.py')):`<br>`    compile(p.read_text(), str(p), 'exec')`<br>`print('syntax ok')`<br>`PY` | 输出 `syntax ok` | 不需要 `torch` 或 `gsplat`，只检查语法。 |
| `spherical_harmonics.py` | `python - <<'PY'`<br>`import torch`<br>`from modules.spherical_harmonics import rgb_to_sh, sh_to_rgb, num_sh_bases`<br>`rgb = torch.tensor([[0.2, 0.4, 0.6]])`<br>`print(num_sh_bases(3), sh_to_rgb(rgb_to_sh(rgb)))`<br>`PY` | 输出 16，并且还原后的 RGB 接近输入 | 需要 `torch`。 |
| `gaussian_model.py` | `python - <<'PY'`<br>`import numpy as np`<br>`from modules.gaussian_model import GaussianModel`<br>`m = GaussianModel.from_point_cloud(np.random.rand(8,3), np.full((8,3),128,dtype=np.uint8), device='cpu')`<br>`print(m.num_gaussians, m.colors.shape)`<br>`PY` | 输出 `8` 和 SH 颜色张量形状 | 需要 `torch`；用 CPU 即可。 |
| `camera.py` | `python - <<'PY'`<br>`import torch`<br>`from modules.camera import Camera`<br>`cam = Camera(640,480,500,500,320,240,torch.eye(4))`<br>`print(cam.viewmat.shape, cam.K)`<br>`PY` | 输出 `(4,4)` 和 `(3,3)` 内参矩阵 | 需要 `torch`；用 CPU 即可。 |
| `losses.py` | `python - <<'PY'`<br>`import torch`<br>`from modules.losses import photometric_loss`<br>`pred = torch.rand(3,32,32)`<br>`gt = torch.rand(3,32,32)`<br>`loss, parts = photometric_loss(pred, gt)`<br>`print(loss.item(), parts.keys())`<br>`PY` | 输出 loss 数值和 `l1/ssim/dssim/total` | 需要 `torch`。 |
| `medium_field.py` | `python - <<'PY'`<br>`import torch`<br>`from modules.medium_field import MediumField, MediumFieldConfig`<br>`m = MediumField(MediumFieldConfig((0,0,0), 1.0))`<br>`beta, sigma = m(torch.zeros(4,3))`<br>`print(beta.shape, sigma.shape, m.medium_rgb.shape)`<br>`PY` | 输出 `(4,3)`、`(4,3)` 和 `(3,)` | 需要 `torch`；用 CPU 即可。 |
| `optim.py` | `python - <<'PY'`<br>`import numpy as np`<br>`from modules.gaussian_model import GaussianModel`<br>`from modules.optim import build_3dgs_optimizer`<br>`m = GaussianModel.from_point_cloud(np.random.rand(4,3), np.ones((4,3)), device='cpu')`<br>`opt = build_3dgs_optimizer(m)`<br>`print([g['name'] for g in opt.param_groups])`<br>`PY` | 输出 6 个 3DGS 参数组名称 | 需要 `torch`；用 CPU 即可。 |
| `renderer.py` | `python - <<'PY'`<br>`import numpy as np, torch`<br>`from modules import Camera, GaussianModel, GaussianRenderer`<br>`m = GaussianModel.from_point_cloud(np.random.rand(16,3), np.ones((16,3)), device='cuda')`<br>`cam = Camera(64,64,50,50,32,32,torch.eye(4,device='cuda'))`<br>`out = GaussianRenderer().render(m, cam)`<br>`print(out.image.shape, out.alpha.shape)`<br>`PY` | 输出 RGB 和 alpha shape | 需要 `torch`、`gsplat`、CUDA。当前环境缺依赖时会失败。 |
| `densification.py` | `python - <<'PY'`<br>`from modules.densification import DensificationConfig, DensificationController`<br>`ctrl = DensificationController(DensificationConfig(interval=10))`<br>`print(ctrl.config.interval)`<br>`PY` | 输出 `10` | 只测配置构造；完整 update 需要训练后有 `RenderOutput` 和 optimizer。 |

外部代码可以自行组合这些模块来实现标准 3DGS 或扩展方法。
