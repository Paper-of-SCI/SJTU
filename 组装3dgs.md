# 组装 3DGS 训练 Curasao

目标：把仓库里已经拆好的 `utils/` 和 `modules/` 组装成一个可运行的 Curasao 3D Gaussian Splatting 训练、评估、渲染流程。

数据目录：

```text
src/datasets/SeathruNeRF_dataset/Curasao
```

当前可运行入口在：

```text
methods/3dgs/train_3dgs_curasao.py
methods/3dgs/render_3dgs_curasao.py
methods/3dgs/render_3dgs_path.py
```

## 0. 先看总调用链

训练主线从 `methods/3dgs/train_3dgs_curasao.py` 进入：

```text
parse_args()
  -> main()
    -> resolve_input_path()
    -> resolve_output_path()
    -> load_colmap_dataset()                  # utils/dataset_loaders.py
    -> GaussianModel.from_point_cloud()       # modules/gaussian_model.py
    -> GaussianRenderer()                     # modules/renderer.py
    -> build_3dgs_optimizer()                 # modules/optim.py
    -> exponential_lr()                       # modules/optim.py
    -> DensificationController()              # modules/densification.py
    -> build_cameras()
      -> Camera.from_scene_data()             # modules/camera.py
    -> for step in training loop:
      -> set_group_lr()                       # modules/optim.py
      -> GaussianRenderer.render()            # modules/renderer.py
        -> GaussianModel.activated_tensors()  # modules/gaussian_model.py
        -> gsplat.rasterization()
      -> photometric_loss()                   # modules/losses.py
      -> loss.backward()
      -> optimizer.step()
      -> DensificationController.update()     # modules/densification.py
      -> compute_psnr()                       # utils/image_utils.py
      -> evaluate_psnr()
      -> save_preview()
        -> save_image()                       # utils/image_utils.py
      -> save_checkpoint()
        -> gaussians_to_ply_dict()            # utils/ply_io.py
        -> write_ply()                        # utils/ply_io.py
```

训练结束后看结果有两条线：

```text
methods/3dgs/render_3dgs_curasao.py
  -> 用数据集里的 train/test/val 相机渲染，能和 GT 算 PSNR

methods/3dgs/render_3dgs_path.py
  -> 构造新视角路径渲染，只输出 novel-view 图片
```

## 1. 先确认底层模块没坏

文件：`test.ipynb`

用途：确认数据读取、相机、PLY、GaussianModel、loss、optimizer、densification、renderer 这些底层积木能单独工作。

运行：

```bash
jupyter notebook test.ipynb
```

在 notebook 里 `Restart Kernel` 后 `Run All`，最后应该看到：

```text
全部必测项通过 / All required tests passed
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

## 2. 第一步：训练脚本入口

文件：`methods/3dgs/train_3dgs_curasao.py`

先看这两个函数：

```python
def parse_args() -> argparse.Namespace:
    ...

def main() -> None:
    ...
```

`parse_args()` 只负责命令行参数，例如：

```text
--data
--out
--iterations
--factor
--holdout
--sh-degree
--lambda-dssim
--save-every
--log-every
--eval-every
--densify-grad-threshold
--disable-densification
```

真正的组装从 `main()` 开始。`main()` 做这些事情：

```text
检查 CUDA
  -> 设置随机种子
  -> 解析输入输出目录
  -> 加载 train/test 数据
  -> 用 COLMAP 点云初始化 GaussianModel
  -> 创建 renderer
  -> 创建 optimizer 和位置学习率 schedule
  -> 创建 densifier
  -> 把 SceneData 转成 Camera 列表
  -> 进入训练循环
  -> 保存 final.ply
```

所以你要改训练流程，优先看 `main()`；要改某个模块的内部行为，再跳到它调用的 `modules/` 或 `utils/` 文件。

## 3. 第二步：解析输入输出路径

文件：`methods/3dgs/train_3dgs_curasao.py`

函数：

```python
def resolve_input_path(path: str) -> Path:
    ...

def resolve_output_path(path: str) -> Path:
    ...
```

调用位置在 `main()`：

```python
data_dir = resolve_input_path(args.data)
out_dir = resolve_output_path(args.out)
preview_dir = out_dir / "previews"
checkpoint_dir = out_dir / "checkpoints"
```

作用：

- `resolve_input_path()`：支持直接传绝对路径，也支持传仓库相对路径。
- `resolve_output_path()`：输出目录如果不是绝对路径，就放到仓库根目录下面。
- `preview_dir`：保存训练中间预览图。
- `checkpoint_dir`：保存中间 PLY checkpoint。

输出目录结构会变成：

```text
outputs/curasao_3dgs/
  previews/
  checkpoints/
  final.ply
```

## 4. 第三步：加载 Curasao 数据

训练脚本调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
scene = load_colmap_dataset(
    str(data_dir),
    split="train",
    load_images=False,
    factor=args.factor,
    holdout=args.holdout,
    opengl=False,
)

test_scene = load_colmap_dataset(
    str(data_dir),
    split="test",
    load_images=False,
    factor=args.factor,
    holdout=args.holdout,
    opengl=False,
)
```

真正实现：

文件：`utils/dataset_loaders.py`

函数：

```python
def load_colmap_dataset(
    data_dir: str,
    split: str = "train",
    load_images: bool = True,
    factor: int = 1,
    holdout: int = 8,
    opengl: bool = True,
) -> SceneData:
    ...
```

`load_colmap_dataset()` 内部继续调用：

```text
read_colmap_model()       # utils/colmap_reader.py，读取 sparse/0
_split_indices()          # utils/dataset_loaders.py，划分 train/test
_find_image_dir()         # utils/dataset_loaders.py，找 images_4/images_wb/images
colmap_image_to_c2w()     # utils/camera_utils.py，把 COLMAP qvec/tvec 转 c2w
_estimate_near_far()      # utils/dataset_loaders.py，根据点云估 near/far
estimate_scene_extent()   # utils/camera_utils.py，估计场景尺度
```

返回的数据类型：

文件：`utils/dataset_loaders.py`

```python
@dataclass(frozen=True)
class SceneData:
    image_paths: list[str]
    images: Optional[np.ndarray]
    c2w_matrices: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    width: int
    height: int
    near: float
    far: float
    scene_extent: float
    point_cloud_xyz: Optional[np.ndarray] = None
    point_cloud_rgb: Optional[np.ndarray] = None
```

这里要注意 `opengl=False`。当前 `gsplat` 训练链路按 COLMAP/OpenCV 相机坐标投影；如果这里错用 OpenGL 坐标，初始点云可能投到相机后面，训练会只看到背景或直接崩。

这一阶段的输入输出：

```text
输入：
  src/datasets/SeathruNeRF_dataset/Curasao/sparse/0
  src/datasets/SeathruNeRF_dataset/Curasao/images_4 或 images_wb/images

输出：
  scene.image_paths
  scene.c2w_matrices
  scene.fx/fy/cx/cy
  scene.width/height
  scene.point_cloud_xyz
  scene.point_cloud_rgb
```

下一步会把：

```text
scene.point_cloud_xyz/rgb -> GaussianModel.from_point_cloud()
scene -> build_cameras() -> Camera.from_scene_data()
```

## 5. 第四步：用 COLMAP 点云初始化 GaussianModel

训练脚本调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
model = GaussianModel.from_point_cloud(
    scene.point_cloud_xyz,
    scene.point_cloud_rgb,
    sh_degree=args.sh_degree,
    device=device,
)
```

真正实现：

文件：`modules/gaussian_model.py`

类和函数：

```python
class GaussianModel(nn.Module):
    ...

@classmethod
def from_point_cloud(
    cls,
    xyz: np.ndarray | Tensor,
    rgb: np.ndarray | Tensor,
    sh_degree: int = 3,
    device: str | torch.device = "cuda",
    initial_opacity: float = 0.1,
    knn: int = 3,
) -> "GaussianModel":
    ...
```

`from_point_cloud()` 内部做这些事：

```text
_to_float_tensor()             # xyz -> torch.float32
_to_rgb_tensor()               # rgb -> [0,1] float tensor
_mean_neighbor_distance()      # 用近邻距离初始化 Gaussian scale
rgb_to_sh()                    # modules/spherical_harmonics.py，RGB 转 0 阶 SH
replace_tensors()              # 替换模型参数，并重置 gradient buffer
```

模型里真正被优化的参数是：

```text
means             # Gaussian 中心，shape (N, 3)
log_scales        # log 尺度，shape (N, 3)
quats             # 旋转四元数，shape (N, 4)
logit_opacities   # opacity 的 logit，shape (N, 1)
features_dc       # 0 阶 SH 颜色，shape (N, 1, 3)
features_rest     # 高阶 SH 颜色，shape (N, K, 3)
```

渲染时不会直接用 raw 参数，而是走：

文件：`modules/gaussian_model.py`

```python
def activated_tensors(self) -> GaussianTensors:
    ...
```

它会输出：

```text
means
scales = exp(log_scales)
quats = normalize(quats)
opacities = sigmoid(logit_opacities)
colors = concat(features_dc, features_rest)
```

下一步会把 `model` 传给：

```text
build_3dgs_optimizer(model, ...)
GaussianRenderer.render(model, camera)
DensificationController.update(model, ...)
save_checkpoint(..., model)
```

## 6. 第五步：创建 renderer

训练脚本调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
renderer = GaussianRenderer(background=(1.0, 1.0, 1.0))
```

真正实现：

文件：`modules/renderer.py`

类：

```python
class GaussianRenderer:
    ...
```

核心函数：

```python
def render(
    self,
    model: GaussianModel,
    camera: Camera,
    sh_degree: Optional[int] = None,
    render_mode: RenderMode = "RGB",
) -> RenderOutput:
    ...
```

`GaussianRenderer.render()` 内部调用：

```text
model.activated_tensors()      # modules/gaussian_model.py
camera.viewmat                 # modules/camera.py，world-to-camera
camera.K                       # modules/camera.py，相机内参矩阵
gsplat.rasterization()         # 真正的 CUDA rasterizer
```

返回：

文件：`modules/renderer.py`

```python
@dataclass
class RenderOutput:
    image: Tensor
    alpha: Tensor
    depth: Optional[Tensor]
    radii: Optional[Tensor]
    means2d: Optional[Tensor]
    metadata: Dict[str, Any]
```

训练循环里会用：

```text
render.image      -> photometric_loss()
render.means2d    -> DensificationController.update()
render.radii      -> DensificationController.update()
render.metadata   -> 取 gaussian_ids，给 packed gsplat 梯度回填用
```

## 7. 第六步：创建 optimizer 和位置学习率 schedule

训练脚本调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
optimizer = build_3dgs_optimizer(model, OptimConfig())
position_lr = exponential_lr(
    1.6e-4,
    1.6e-6,
    max_steps=args.iterations,
    delay_steps=1000,
    delay_mult=0.01,
)
```

真正实现：

文件：`modules/optim.py`

```python
@dataclass
class OptimConfig:
    position_lr: float = 1.6e-4
    feature_lr: float = 2.5e-3
    opacity_lr: float = 5.0e-2
    scaling_lr: float = 5.0e-3
    rotation_lr: float = 1.0e-3
    eps: float = 1.0e-15
```

```python
def build_3dgs_optimizer(
    model: GaussianModel,
    config: OptimConfig | None = None,
) -> torch.optim.Adam:
    ...
```

`build_3dgs_optimizer()` 会创建 6 个参数组：

```text
name="means"             -> model.means
name="features_dc"       -> model.features_dc
name="features_rest"     -> model.features_rest
name="logit_opacities"   -> model.logit_opacities
name="log_scales"        -> model.log_scales
name="quats"             -> model.quats
```

训练循环里每一步会调用：

文件：`modules/optim.py`

```python
def set_group_lr(optimizer: torch.optim.Optimizer, group_name: str, lr: float) -> None:
    ...
```

调用位置：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
set_group_lr(optimizer, "means", position_lr(step))
```

意思是只有 `means` 位置学习率按 3DGS 风格衰减，其他参数组保持 `OptimConfig` 里的固定学习率。

## 8. 第七步：创建 densification 控制器

训练脚本调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
densifier = DensificationController(
    DensificationConfig(
        start_step=500,
        stop_step=max(args.iterations - 500, 501),
        interval=100,
        grad_threshold=args.densify_grad_threshold,
        scene_extent=float(scene.scene_extent),
        percent_dense=0.01,
        min_opacity=0.005,
        opacity_reset_interval=3000,
    )
)
```

真正实现：

文件：`modules/densification.py`

```python
@dataclass
class DensificationConfig:
    start_step: int = 500
    stop_step: int = 15_000
    interval: int = 100
    grad_threshold: float = 2.0e-4
    scene_extent: float = 1.0
    percent_dense: float = 0.01
    min_opacity: float = 0.005
    max_screen_radius: Optional[float] = None
    num_splits: int = 2
    opacity_reset_interval: int = 3000
    reset_opacity: float = 0.01
    use_absgrad: bool = True
```

```python
class DensificationController:
    def update(
        self,
        model: GaussianModel,
        render_output: RenderOutput,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> DensificationStats:
        ...
```

`DensificationController.update()` 的职责：

```text
读取 render_output.means2d 的屏幕空间梯度
  -> 累积到 model.gradient_accum / model.gradient_count
  -> 到达 interval 时调用 _densify()
  -> clone 小尺度高梯度 Gaussian
  -> split 大尺度高梯度 Gaussian
  -> prune 低 opacity Gaussian
  -> 同步修补 Adam optimizer state
  -> 必要时 reset opacity
```

关键点：当前 renderer 默认 `packed=True`，所以 `means2d` 和 `radii` 不一定是完整 `N` 个 Gaussian 的 dense 数组。`update()` 会把：

```python
indices=render_output.metadata.get("gaussian_ids")
```

传给：

文件：`modules/gaussian_model.py`

```python
def accumulate_gradient_stats(
    self,
    means2d: Tensor,
    visibility: Optional[Tensor] = None,
    use_absgrad: bool = True,
    indices: Optional[Tensor] = None,
) -> None:
    ...
```

这一步就是把 packed rasterizer 里可见 Gaussian 的梯度，回填到对应的全局 Gaussian id 上。

## 9. 第八步：把 SceneData 转成 Camera

训练脚本调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
train_cameras = build_cameras(scene, device)
test_cameras = build_cameras(test_scene, device)
```

本地辅助函数：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
def build_cameras(scene, device: torch.device) -> list[Camera]:
    return [
        Camera.from_scene_data(scene, index, device=device, load_image=True)
        for index in range(len(scene.image_paths))
    ]
```

真正实现：

文件：`modules/camera.py`

```python
@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: Tensor
    image: Optional[Tensor] = None
    image_path: str = ""
    near: float = 0.01
    far: float = 1.0e10
```

构造函数：

```python
@classmethod
def from_scene_data(
    cls,
    scene,
    index: int,
    device: str | torch.device = "cuda",
    load_image: bool = True,
) -> "Camera":
    ...
```

`Camera.from_scene_data()` 会做：

```text
从 scene.fx/fy/cx/cy 取第 index 张相机内参
从 scene.c2w_matrices 取第 index 张 c2w
如果 load_image=True，读取 GT 图像
把 GT 图像转成 torch tensor，shape 是 (3, H, W)
```

renderer 会用 `Camera` 的两个属性：

文件：`modules/camera.py`

```python
@property
def viewmat(self) -> Tensor:
    return torch.linalg.inv(self.c2w)

@property
def K(self) -> Tensor:
    return torch.tensor(
        [[self.fx, 0.0, self.cx],
         [0.0, self.fy, self.cy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=self.device,
    )
```

所以组装关系是：

```text
SceneData
  -> Camera.from_scene_data()
    -> Camera.viewmat
    -> Camera.K
      -> GaussianRenderer.render()
```

## 10. 第九步：训练循环逐行组装

核心循环在：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
for step in progress:
    step_started_at = time.perf_counter()
    camera_index = random.choice(camera_indices)
    camera = train_cameras[camera_index]

    optimizer.zero_grad(set_to_none=True)
    set_group_lr(optimizer, "means", position_lr(step))
    render = renderer.render(model, camera)
    loss, parts = photometric_loss(
        render.image.clamp(0.0, 1.0),
        camera.image,
        lambda_dssim=args.lambda_dssim,
    )
    loss.backward()
    optimizer.step()

    stats = None
    if not args.disable_densification:
        stats = densifier.update(model, render, optimizer, step)
```

这段的调用顺序不能乱：

```text
1. random.choice(camera_indices)
   -> 随机选一张训练视角

2. optimizer.zero_grad(set_to_none=True)
   -> 清上一轮梯度

3. set_group_lr(optimizer, "means", position_lr(step))
   -> 更新 Gaussian 中心点学习率

4. renderer.render(model, camera)
   -> 用当前 Gaussian 参数渲染当前相机

5. photometric_loss(render.image, camera.image)
   -> 计算渲染图和 GT 图之间的 L1 + DSSIM

6. loss.backward()
   -> 梯度从图像误差反传到 Gaussian 参数和 means2d

7. optimizer.step()
   -> Adam 更新 Gaussian 参数

8. densifier.update(model, render, optimizer, step)
   -> 用本轮反传得到的屏幕空间梯度做 clone/split/prune
```

为什么 `densifier.update()` 放在 `optimizer.step()` 后面：

```text
本轮梯度先用于参数更新
  -> densification 再根据这些梯度调整下一轮的 Gaussian 结构
```

如果先 densify 再 `optimizer.step()`，参数数量和 optimizer state 更容易错位。

## 11. 第十步：渲染函数内部到底调用什么

训练循环调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
render = renderer.render(model, camera)
```

进入：

文件：`modules/renderer.py`

```python
def render(self, model: GaussianModel, camera: Camera, ...):
    tensors = model.activated_tensors()
    background = _background_tensor(self.background, camera.device)
    colors, alphas, meta = rasterization(
        means=tensors.means,
        quats=tensors.quats,
        scales=tensors.scales,
        opacities=tensors.opacities,
        colors=tensors.colors,
        viewmats=camera.viewmat[None, ...],
        Ks=camera.K[None, ...],
        width=camera.width,
        height=camera.height,
        near_plane=camera.near,
        far_plane=camera.far,
        sh_degree=degree,
        packed=self.packed,
        render_mode=render_mode,
        sparse_grad=self.sparse_grad,
        absgrad=self.absgrad,
        rasterize_mode=self.rasterize_mode,
    )
```

这里的输入来源：

```text
tensors.means       <- GaussianModel.means
tensors.quats       <- normalize(GaussianModel.quats)
tensors.scales      <- exp(GaussianModel.log_scales)
tensors.opacities   <- sigmoid(GaussianModel.logit_opacities)
tensors.colors      <- features_dc + features_rest
camera.viewmat      <- inverse(camera.c2w)
camera.K            <- fx/fy/cx/cy
camera.width/height <- SceneData 分辨率
```

`gsplat.rasterization()` 输出后，`render()` 会整理成训练更方便的 CHW 格式：

```text
colors -> render.image，shape (3, H, W)
alphas -> render.alpha，shape (1, H, W)
meta["radii"] -> render.radii
meta["means2d"] -> render.means2d
meta -> render.metadata
```

## 12. 第十一步：loss 是哪个文件哪个函数

训练循环调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
loss, parts = photometric_loss(
    render.image.clamp(0.0, 1.0),
    camera.image,
    lambda_dssim=args.lambda_dssim,
)
```

真正实现：

文件：`modules/losses.py`

```python
def photometric_loss(
    pred: Tensor,
    gt: Tensor,
    lambda_dssim: float = 0.2,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    l1 = l1_loss(pred, gt)
    ssim_value = ssim(pred, gt)
    dssim = (1.0 - ssim_value) * 0.5
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim
    return total, {"l1": l1, "ssim": ssim_value, "dssim": dssim, "total": total}
```

`photometric_loss()` 内部调用：

```text
l1_loss()   # modules/losses.py
ssim()      # modules/losses.py，torch conv2d 实现，可反传
```

输出：

```text
loss          -> loss.backward()
parts["l1"]   -> 打印日志
parts["ssim"] -> 打印日志
```

## 13. 第十二步：densification 内部怎么改 Gaussian 数量

训练循环调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
stats = densifier.update(model, render, optimizer, step)
```

进入：

文件：`modules/densification.py`

```python
def update(self, model, render_output, optimizer, step) -> DensificationStats:
    visibility = _visibility_from_output(render_output, model.num_gaussians)
    if render_output.means2d is not None:
        model.accumulate_gradient_stats(
            render_output.means2d,
            visibility=visibility,
            use_absgrad=self.config.use_absgrad,
            indices=render_output.metadata.get("gaussian_ids"),
        )
    ...
```

到达 densify 间隔后：

文件：`modules/densification.py`

```python
def _densify(self, model, optimizer, render_output) -> DensificationStats:
    avg_grad = model.gradient_accum / model.gradient_count.clamp_min(1.0)
    high_grad = avg_grad >= self.config.grad_threshold
    max_scale = model.scales.detach().max(dim=-1).values
    dense_threshold = self.config.scene_extent * self.config.percent_dense
    clone_mask = high_grad & (max_scale <= dense_threshold)
    split_mask = high_grad & (max_scale > dense_threshold)
    ...
```

真正改变 Gaussian 数量的是：

文件：`modules/gaussian_model.py`

```python
def clone(self, mask: Tensor) -> int:
    ...

def split(self, mask: Tensor, num_splits: int = 2, scale_shrink: float = 1.6) -> tuple[int, Tensor]:
    ...

def prune(self, remove_mask: Tensor) -> Tensor:
    ...
```

同步 optimizer state 的函数在：

文件：`modules/densification.py`

```text
_snapshot_optimizer()
_patch_optimizer_append()
_patch_optimizer_append_and_prune()
_patch_optimizer_prune()
_replace_group_param()
_zero_optimizer_state_for()
```

所以 densification 的组装链是：

```text
RenderOutput.means2d / metadata["gaussian_ids"]
  -> GaussianModel.accumulate_gradient_stats()
  -> DensificationController._densify()
  -> GaussianModel.clone() / split() / prune()
  -> _patch_optimizer_*()
  -> DensificationStats
```

`DensificationStats` 会回到训练脚本，用来打印：

```text
clone
split
prune
total
high_grad
grad_max
opacity_reset
```

## 14. 第十三步：训练中评估 PSNR

训练脚本调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
psnr = compute_psnr(
    render.image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(),
    camera.image.detach().permute(1, 2, 0).cpu().numpy(),
)
```

真正实现：

文件：`utils/image_utils.py`

```python
def compute_psnr(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    ...
```

测试集评估走本地辅助函数：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
@torch.no_grad()
def evaluate_psnr(
    model: GaussianModel,
    renderer: GaussianRenderer,
    cameras: list[Camera],
) -> float:
    ...
```

调用位置：

```python
if args.eval_every > 0 and (step == 1 or step % args.eval_every == 0 or step == args.iterations):
    eval_psnr = evaluate_psnr(model, renderer, test_cameras)
```

评估链路：

```text
test_cameras
  -> renderer.render(model, camera)
  -> compute_psnr(render.image, camera.image)
  -> mean PSNR
```

这个评估不更新模型，因为 `evaluate_psnr()` 加了 `@torch.no_grad()`。

## 15. 第十四步：保存 preview 和 PLY checkpoint

训练脚本调用：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
if step == 1 or step % args.save_every == 0 or step == args.iterations:
    save_preview(preview_dir / f"step_{step:06d}.png", render.image)
    save_checkpoint(checkpoint_dir / f"step_{step:06d}.ply", model)

save_checkpoint(out_dir / "final.ply", model)
```

本地辅助函数：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
def save_preview(path: Path, image: torch.Tensor) -> None:
    array = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    save_image(str(path), array)
```

`save_preview()` 调用：

文件：`utils/image_utils.py`

```python
def save_image(path: str, image: np.ndarray, quality: int = 95) -> None:
    ...
```

保存 PLY 的辅助函数：

文件：`methods/3dgs/train_3dgs_curasao.py`

```python
def save_checkpoint(path: Path, model: GaussianModel) -> None:
    with torch.no_grad():
        data = gaussians_to_ply_dict(
            model.means.detach().cpu().numpy(),
            model.log_scales.detach().cpu().numpy(),
            model.quats.detach().cpu().numpy(),
            model.logit_opacities.detach().cpu().numpy(),
            model.features_dc.detach().cpu().numpy(),
            model.features_rest.detach().cpu().numpy(),
        )
    write_ply(str(path), data)
```

真正实现：

文件：`utils/ply_io.py`

```python
def gaussians_to_ply_dict(...):
    ...

def write_ply(path: str, data: Dict[str, np.ndarray]) -> None:
    ...
```

保存链路：

```text
GaussianModel raw tensors
  -> gaussians_to_ply_dict()
  -> write_ply()
  -> outputs/curasao_3dgs/checkpoints/step_XXXXXX.ply
  -> outputs/curasao_3dgs/final.ply
```

注意：这里的 `.ply` 是 3DGS Gaussian 参数 checkpoint，不是三角网格，也不是普通点云可视化文件。里面存的是：

```text
x/y/z
f_dc_*
f_rest_*
opacity
scale_*
rot_*
```

## 16. 先跑 smoke test

先用很少迭代确认完整链路能跑通：

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

这个命令会验证：

```text
load_colmap_dataset()
  -> GaussianModel.from_point_cloud()
  -> build_cameras()
  -> GaussianRenderer.render()
  -> photometric_loss()
  -> loss.backward()
  -> optimizer.step()
  -> save_preview()
  -> save_checkpoint()
```

输出应该至少包含：

```text
outputs/curasao_smoke/previews/step_000001.png
outputs/curasao_smoke/previews/step_000005.png
outputs/curasao_smoke/checkpoints/step_000001.ply
outputs/curasao_smoke/checkpoints/step_000005.ply
outputs/curasao_smoke/final.ply
```

## 17. 正式训练 Curasao

确认 smoke test 没问题后，跑正式训练：

```bash
python methods/3dgs/train_3dgs_curasao.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao \
  --out outputs/curasao_3dgs \
  --iterations 7000 \
  --factor 4 \
  --densify-grad-threshold 2e-5 \
  --save-every 1000 \
  --log-every 50
```

如果显存不够，把分辨率降下来：

```bash
--factor 8
```

如果只想先看普通优化，不想让 Gaussian 数量变化：

```bash
--disable-densification
```

如果测试集评估太频繁影响速度：

```bash
--eval-every 1000
```

## 18. 看训练是否正常

脚本会打印类似：

```text
设备：CUDA GPU='NVIDIA GeForce RTX 3070 Laptop GPU' 显存=8.00GB
数据集：/home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset/Curasao
数据划分：训练=18 张，测试=3 张，holdout=8，分辨率=444x295，降采样 factor=4
初始 Gaussian 数量：...，训练步数：7000，启用致密化：True，densify_grad_threshold=2e-05
训练进度: ... loss=... 训练PSNR=... G=...
第 000700 步 | loss=... | L1=... | SSIM=... | 训练PSNR=... | Gaussian=...
测试评估 | 第 001000 步 | 测试图像=3 | 测试PSNR=...
```

主要看这些：

```text
设备：CUDA GPU=...
  -> 确认在用 GPU

数据划分：训练=...，测试=...
  -> 确认 holdout 生效

loss
  -> 正常情况下整体下降

训练PSNR
  -> 正常情况下整体上升

测试PSNR
  -> 用测试集看泛化，由 --eval-every 控制频率

Gaussian=...
  -> densification 开启后数量会 clone/split/prune

outputs/.../previews
  -> 图像应该逐渐接近 GT
```

GPU 利用率只有 30% 多不一定代表没用 GPU。常见原因是：

```text
factor 太大，分辨率低
当前训练是单视角逐步优化
Gaussian 初始数量还不够大
测试评估和保存 checkpoint 会触发 GPU/CPU 同步
```

正式训练建议优先用 `--factor 4`；如果显存不够再调到 `--factor 8`。

## 19. 训练完用数据集相机看结果

入口文件：

```text
methods/3dgs/render_3dgs_curasao.py
```

这个脚本适合回答：训练出来的 `final.ply` 在 Curasao 的真实相机上渲染得怎么样。

主调用链：

```text
parse_args()
  -> main()
    -> resolve_input_path()
    -> resolve_output_path()
    -> load_colmap_dataset()              # utils/dataset_loaders.py
    -> load_gaussian_checkpoint()
      -> read_ply()                       # utils/ply_io.py
      -> ply_dict_to_gaussians()          # utils/ply_io.py
      -> GaussianModel.replace_tensors()  # modules/gaussian_model.py
    -> GaussianRenderer()
    -> for index in range(count):
      -> Camera.from_scene_data()         # modules/camera.py
      -> renderer.render()
      -> compute_psnr()                   # utils/image_utils.py
      -> save_image()                     # utils/image_utils.py
```

运行：

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

只看一张：

```bash
python methods/3dgs/render_3dgs_curasao.py \
  --checkpoint outputs/curasao_3dgs/final.ply \
  --out outputs/curasao_3dgs/renders_one \
  --split test \
  --factor 4 \
  --max-images 1
```

关键函数说明：

文件：`methods/3dgs/render_3dgs_curasao.py`

```python
def load_gaussian_checkpoint(path: Path, device: torch.device) -> GaussianModel:
    means, log_scales, quats, logit_opacities, features_dc, features_rest = ply_dict_to_gaussians(read_ply(str(path)))
    sh_bases = 1 + features_rest.shape[1]
    sh_degree = int(round(sh_bases**0.5 - 1))
    model = GaussianModel(sh_degree=sh_degree).to(device)
    model.replace_tensors(...)
    return model
```

意思是：

```text
final.ply
  -> read_ply()
  -> ply_dict_to_gaussians()
  -> GaussianModel.replace_tensors()
  -> renderer.render()
```

## 20. 训练完渲染新视角路径

入口文件：

```text
methods/3dgs/render_3dgs_path.py
```

这个脚本适合回答：不用数据集原始相机，自己构造一条相机路径看 novel view。

主调用链：

```text
parse_args()
  -> main()
    -> resolve_input_path()
    -> resolve_output_path()
    -> load_colmap_dataset()             # utils/dataset_loaders.py
    -> load_gaussian_checkpoint()
      -> read_ply()                      # utils/ply_io.py
      -> ply_dict_to_gaussians()         # utils/ply_io.py
      -> GaussianModel.replace_tensors() # modules/gaussian_model.py
    -> GaussianRenderer()
    -> build_path_cameras()
      -> interpolate_pose() + make_camera()
      或 build_orbit_cameras()
        -> look_at_c2w()
        -> make_camera()
    -> for camera in cameras:
      -> renderer.render()
      -> save_image()
```

插值路径：

```bash
python methods/3dgs/render_3dgs_path.py \
  --checkpoint outputs/curasao_3dgs/final.ply \
  --out outputs/curasao_3dgs/path_interpolate \
  --mode interpolate \
  --split train \
  --factor 4 \
  --frames 60 \
  --start 0 \
  --end -1
```

环绕路径：

```bash
python methods/3dgs/render_3dgs_path.py \
  --checkpoint outputs/curasao_3dgs/final.ply \
  --out outputs/curasao_3dgs/path_orbit \
  --mode orbit \
  --split train \
  --factor 4 \
  --frames 60 \
  --radius-scale 1.0
```

关键函数：

文件：`methods/3dgs/render_3dgs_path.py`

```python
def build_path_cameras(args: argparse.Namespace, scene, device: torch.device) -> list[Camera]:
    ...

def interpolate_pose(c2w_a: np.ndarray, c2w_b: np.ndarray, t: float) -> np.ndarray:
    ...

def build_orbit_cameras(args: argparse.Namespace, scene, device: torch.device) -> list[Camera]:
    ...

def look_at_c2w(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    ...

def make_camera(scene, c2w: np.ndarray, device: torch.device, image_path: str) -> Camera:
    ...
```

`interpolate` 模式：

```text
取 scene.c2w_matrices[start]
取 scene.c2w_matrices[end]
  -> interpolate_pose()
    -> rotmat_to_quat()
    -> slerp()
    -> quat_to_rotmat()
  -> make_camera()
```

`orbit` 模式：

```text
用所有相机中心估 scene center
  -> 生成一圈相机位置
  -> look_at_c2w(position, center, up)
  -> make_camera()
```

输出：

```text
outputs/curasao_3dgs/path_interpolate/frame_0000.png
outputs/curasao_3dgs/path_interpolate/frame_0001.png
...
```

## 21. 如果要自己改组装逻辑，应该改哪里

常见改动位置：

```text
换数据读取方式
  -> utils/dataset_loaders.py
  -> methods/3dgs/train_3dgs_curasao.py 里的 load_colmap_dataset() 调用

换相机定义或坐标系
  -> modules/camera.py
  -> utils/camera_utils.py
  -> 注意 train/render 两边必须一致

换 Gaussian 初始化
  -> modules/gaussian_model.py 的 GaussianModel.from_point_cloud()

换渲染参数
  -> modules/renderer.py 的 GaussianRenderer.__init__() / render()

换 loss
  -> modules/losses.py 的 photometric_loss()

换优化器学习率
  -> modules/optim.py 的 OptimConfig / build_3dgs_optimizer()
  -> methods/3dgs/train_3dgs_curasao.py 里的 exponential_lr()

换 densification 策略
  -> modules/densification.py
  -> methods/3dgs/train_3dgs_curasao.py 里的 DensificationConfig()

换 checkpoint 格式
  -> utils/ply_io.py
  -> train/render 两边的 save_checkpoint() / load_gaussian_checkpoint() 必须一起改

换 novel-view 路径
  -> methods/3dgs/render_3dgs_path.py
```

不要把这些逻辑都塞回 `train_3dgs_curasao.py`。训练脚本只负责组装和调度；数据、相机、模型、渲染、loss、optimizer、densification、PLY IO 都应该继续保持独立模块。

## 22. 最小闭环总结

如果只记一条完整链路，就是：

```text
utils.dataset_loaders.load_colmap_dataset()
  -> modules.gaussian_model.GaussianModel.from_point_cloud()
  -> modules.camera.Camera.from_scene_data()
  -> modules.renderer.GaussianRenderer.render()
  -> modules.losses.photometric_loss()
  -> modules.optim.build_3dgs_optimizer()
  -> modules.densification.DensificationController.update()
  -> utils.ply_io.gaussians_to_ply_dict()
  -> utils.ply_io.write_ply()
  -> methods/3dgs/render_3dgs_curasao.py 或 render_3dgs_path.py 查看结果
```

对应可执行命令：

```bash
python methods/3dgs/train_3dgs_curasao.py \
  --data src/datasets/SeathruNeRF_dataset/Curasao \
  --out outputs/curasao_3dgs \
  --iterations 7000 \
  --factor 4 \
  --densify-grad-threshold 2e-5 \
  --save-every 1000 \
  --log-every 50

python methods/3dgs/render_3dgs_curasao.py \
  --checkpoint outputs/curasao_3dgs/final.ply \
  --out outputs/curasao_3dgs/renders_test \
  --split test \
  --factor 4

python methods/3dgs/render_3dgs_path.py \
  --checkpoint outputs/curasao_3dgs/final.ply \
  --out outputs/curasao_3dgs/path_interpolate \
  --mode interpolate \
  --split train \
  --factor 4 \
  --frames 60
```
