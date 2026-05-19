# 3DGS 重建完整计算流程

这份文档只讲一件事：3D Gaussian Splatting 到底是怎么从多张图片重建出一个 3D 场景的。

你要按这个顺序理解：

```text
COLMAP 数据
  -> 相机参数
  -> 稀疏点云
  -> 初始化一堆 3D Gaussian
  -> 选一张训练图
  -> 把 3D Gaussian 投影到这张图上
  -> 光栅化合成一张渲染图
  -> 和真实图算 loss
  -> 反传更新 Gaussian 参数
  -> 根据梯度 clone / split / prune
  -> 保存 final.ply
  -> 用 final.ply 从任意相机视角渲染
```

当前仓库对应的训练入口：

```text
methods/3dgs/train_3dgs_scene.py
```

核心模块：

```text
utils/dataset_loaders.py      # 读 COLMAP 数据，得到 SceneData
modules/camera.py             # 把 SceneData 里的某一张图变成 Camera
modules/gaussian_model.py     # 存 Gaussian 参数
modules/renderer.py           # 调 gsplat.rasterization 渲染
modules/losses.py             # 计算 L1 + DSSIM loss
modules/optim.py              # 建 Adam optimizer
modules/densification.py      # clone / split / prune
utils/ply_io.py               # 保存和读取 3DGS PLY
```

## 1. 3DGS 重建到底在优化什么

3DGS 不是直接训练一个神经网络。它优化的是一堆可学习的 3D Gaussian。

假设场景里有 `N` 个 Gaussian。第 `i` 个 Gaussian 有这些参数：

```text
mu_i              3D 中心点，shape (3,)
log_scale_i       3 个方向的 log 尺度，shape (3,)
quat_i            旋转四元数，shape (4,)
opacity_logit_i   透明度 logit，shape (1,)
features_dc_i     0 阶球谐颜色，shape (1, 3)
features_rest_i   高阶球谐颜色，shape (K, 3)
```

仓库里对应：

文件：`modules/gaussian_model.py`

```python
self.means
self.log_scales
self.quats
self.logit_opacities
self.features_dc
self.features_rest
```

训练要做的是：

```text
调这些 Gaussian 参数
  -> 让它们从训练相机渲染出来的图
  -> 尽可能接近真实训练图
```

数学目标可以写成：

```text
min over Gaussian params:
  sum over training cameras c:
    Loss(Render(Gaussians, Camera_c), Image_c)
```

也就是：

```text
让渲染图 R_c 尽量接近真实图 I_c
```

## 2. 输入数据是什么

训练使用 Curasao：

```text
src/datasets/SeathruNeRF_dataset/Curasao
```

里面最关键的是两类数据：

```text
sparse/0/
  cameras.bin 或 cameras.txt
  images.bin 或 images.txt
  points3D.bin 或 points3D.txt

images_4 / images_wb / images
  真实 RGB 图片
```

仓库读取入口：

文件：`utils/dataset_loaders.py`

```python
scene = load_colmap_dataset(
    data_dir,
    split="train",
    load_images=False,
    factor=4,
    holdout=8,
    opengl=False,
)
```

返回的是：

```python
SceneData(
    image_paths,
    images,
    c2w_matrices,
    fx, fy, cx, cy,
    width, height,
    near, far,
    scene_extent,
    point_cloud_xyz,
    point_cloud_rgb,
)
```

你先记住：

```text
SceneData = 图片路径 + 相机位姿 + 相机内参 + 稀疏点云
```

后面的所有训练都从这个 `SceneData` 出发。

## 3. 第一步计算：COLMAP 相机转成 c2w

COLMAP 给的是 world-to-camera：

```text
X_cam = R_cw * X_world + t_cw
```

这里：

```text
X_world  是世界坐标点
X_cam    是相机坐标点
R_cw     是世界到相机的旋转
t_cw     是世界到相机的平移
```

仓库里对应：

文件：`utils/camera_utils.py`

```python
def colmap_image_to_c2w(qvec, tvec, opengl=True):
    rot = qvec_to_rotmat(qvec)
    w2c = eye(4)
    w2c[:3, :3] = rot
    w2c[:3, 3] = tvec
    c2w = inv(w2c)
    if opengl:
        c2w[:3, 1:3] *= -1.0
    return c2w
```

当前训练必须用：

```python
opengl=False
```

原因：这里的 `gsplat` 链路按 COLMAP/OpenCV 坐标工作。COLMAP 相机坐标是：

```text
x 向右
y 向下
z 向前
```

所以从 COLMAP 数据得到：

```text
w2c = [R_cw | t_cw]
c2w = inverse(w2c)
```

之后 `Camera.viewmat` 又会算回：

文件：`modules/camera.py`

```python
@property
def viewmat(self):
    return torch.linalg.inv(self.c2w)
```

所以 renderer 实际拿到的是：

```text
viewmat = w2c
```

## 4. 第二步计算：相机内参 K

每张图都有内参：

```text
fx, fy, cx, cy
```

组成相机矩阵：

```text
K = [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  1 ]]
```

仓库里对应：

文件：`modules/camera.py`

```python
@property
def K(self):
    return torch.tensor(
        [[self.fx, 0.0, self.cx],
         [0.0, self.fy, self.cy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=self.device,
    )
```

如果原图宽高被 `factor=4` 降采样，内参也要一起除以 4：

文件：`utils/dataset_loaders.py`

```python
width = camera.width // factor
height = camera.height // factor
fx = camera.fx / factor
fy = camera.fy / factor
cx = camera.cx / factor
cy = camera.cy / factor
```

这一步很重要。图片缩小了，相机内参也必须缩小，否则投影位置会错。

## 5. 第三步计算：训练集和测试集怎么分

仓库用 `holdout=8`：

文件：`utils/dataset_loaders.py`

```python
def _split_indices(values, split, holdout):
    if split in {"test", "val"}:
        return [value for i, value in enumerate(values) if holdout > 0 and i % holdout == 0]
    return [value for i, value in enumerate(values) if holdout <= 0 or i % holdout != 0]
```

意思是：

```text
第 0, 8, 16, ... 张图 -> test
其他图 -> train
```

训练时：

```text
train 图片用于更新 Gaussian
test 图片只用于评估 PSNR，不更新参数
```

## 6. 第四步计算：用 COLMAP 点云初始化 Gaussian

COLMAP 的 `points3D` 提供：

```text
point_cloud_xyz  shape (N, 3)
point_cloud_rgb  shape (N, 3)
```

训练脚本调用：

文件：`methods/3dgs/train_3dgs_scene.py`

```python
model = GaussianModel.from_point_cloud(
    scene.point_cloud_xyz,
    scene.point_cloud_rgb,
    sh_degree=args.sh_degree,
    device=device,
)
```

真正计算在：

文件：`modules/gaussian_model.py`

```python
def from_point_cloud(xyz, rgb, sh_degree=3, device="cuda", initial_opacity=0.1, knn=3):
    ...
```

对每个点 `p_i`，初始化一个 Gaussian。

### 6.1 中心点 means

最直接：

```text
mu_i = p_i
```

代码对应：

```python
xyz_t = _to_float_tensor(xyz, device)
model.means = xyz_t
```

意思是：

```text
COLMAP 稀疏点在哪里，初始 Gaussian 中心就放在哪里
```

### 6.2 颜色 RGB 转成 0 阶 SH

先把 RGB 变成 `[0,1]`：

```text
rgb01_i = rgb_i / 255
```

然后转 0 阶球谐系数：

文件：`modules/spherical_harmonics.py`

```python
SH_C0 = 0.28209479177387814

def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0
```

所以：

```text
features_dc_i = (rgb01_i - 0.5) / 0.28209479177387814
```

为什么要这样？

因为 3DGS 的颜色一般存成球谐系数。只看 0 阶时，反变换是：

```text
rgb = SH_C0 * features_dc + 0.5
```

所以刚初始化时，0 阶 SH 能还原原始点云颜色。

举例：

```text
RGB = (128, 64, 32)
rgb01 = (0.50196, 0.25098, 0.12549)
features_dc = ((rgb01 - 0.5) / 0.28209479)
            = (0.00695, -0.88275, -1.32760)
```

### 6.3 尺度 scale

每个 Gaussian 不是一个点，而是一个椭球。它有 3 个方向尺度：

```text
scale_i = (sx, sy, sz)
```

仓库里不直接存 `scale`，而是存：

```text
log_scale_i = log(scale_i)
```

原因：优化时 `log_scale` 可以是任意实数，但实际尺度：

```text
scale_i = exp(log_scale_i)
```

永远是正数。

初始化尺度用近邻距离：

文件：`modules/gaussian_model.py`

```python
mean_dist = _mean_neighbor_distance(xyz_t, k=knn).clamp_min(1e-7)
log_scales = mean_dist.log().unsqueeze(-1).expand(n, 3)
```

数学上：

```text
d_i = mean distance from point i to its k nearest neighbors
scale_i = (d_i, d_i, d_i)
log_scale_i = (log d_i, log d_i, log d_i)
```

如果一个点附近很密：

```text
d_i 小 -> 初始 Gaussian 小
```

如果一个点附近很稀：

```text
d_i 大 -> 初始 Gaussian 大
```

### 6.4 旋转 quaternion

初始旋转是单位旋转：

```text
quat_i = (1, 0, 0, 0)
```

仓库使用 wxyz 顺序。

渲染前会归一化：

文件：`modules/gaussian_model.py`

```python
normalized_quats = normalize(self.quats, dim=-1)
```

### 6.5 透明度 opacity

初始化后验透明度是：

```text
opacity_i = 0.1
```

但仓库不直接存 `opacity`，而是存：

```text
opacity_logit_i = log(opacity_i / (1 - opacity_i))
```

所以：

```text
opacity_logit_i = log(0.1 / 0.9) = -2.1972
```

渲染时再变回：

```text
opacity_i = sigmoid(opacity_logit_i)
```

原因同样是为了优化稳定：

```text
opacity_logit 可以是任意实数
sigmoid(opacity_logit) 一定在 0 到 1 之间
```

### 6.6 高阶 SH

如果 `sh_degree=3`，球谐基数量是：

```text
(degree + 1)^2 = 16
```

其中：

```text
features_dc   是第 0 阶，shape (N, 1, 3)
features_rest 是剩下 15 个基，shape (N, 15, 3)
```

初始化：

```text
features_rest = 0
```

训练过程中它们会被 Adam 更新，用来表达视角相关颜色。

## 7. 第五步计算：一个 Gaussian 的真实数学形状

第 `i` 个 Gaussian 的 3D 分布可以理解成：

```text
G_i(x) = exp(-0.5 * (x - mu_i)^T * Sigma_i^-1 * (x - mu_i))
```

其中协方差：

```text
Sigma_i = R_i * diag(scale_i^2) * R_i^T
```

这里：

```text
mu_i       来自 model.means
scale_i    = exp(model.log_scales)
R_i        由 normalized quaternion 转成旋转矩阵
opacity_i  = sigmoid(model.logit_opacities)
color_i    由 SH 系数和观察方向算出来
```

仓库渲染前会把 raw 参数激活：

文件：`modules/gaussian_model.py`

```python
def activated_tensors(self):
    return GaussianTensors(
        means=self.means,
        scales=self.scales,
        quats=self.normalized_quats,
        opacities=self.opacities,
        colors=self.colors,
    )
```

对应计算：

```text
scales = exp(log_scales)
quats = normalize(quats)
opacities = sigmoid(logit_opacities)
colors = concat(features_dc, features_rest)
```

## 8. 第六步计算：把一张训练图变成 Camera

训练脚本先构造所有训练相机：

文件：`methods/3dgs/train_3dgs_scene.py`

```python
train_cameras = build_cameras(scene, device)
```

本地函数：

```python
def build_cameras(scene, device):
    return [Camera.from_scene_data(scene, index, device=device, load_image=True)
            for index in range(len(scene.image_paths))]
```

真正实现：

文件：`modules/camera.py`

```python
Camera.from_scene_data(scene, index, device=device, load_image=True)
```

它做三件事：

```text
1. 取第 index 张图的 fx/fy/cx/cy
2. 取第 index 张图的 c2w
3. 读取 GT 图像，转成 shape (3, H, W)
```

所以一个 `Camera` 包含：

```text
K          相机内参
viewmat    world-to-camera 矩阵
image      真实 GT 图像
width      图像宽
height     图像高
near/far   渲染裁剪范围
```

## 9. 第七步计算：一次训练迭代完整展开

训练循环在：

文件：`methods/3dgs/train_3dgs_scene.py`

```python
for step in progress:
    camera_index = random.choice(camera_indices)
    camera = train_cameras[camera_index]

    optimizer.zero_grad(set_to_none=True)
    set_group_lr(optimizer, "means", position_lr(step))
    render = renderer.render(model, camera)
    loss, parts = photometric_loss(render.image.clamp(0.0, 1.0), camera.image)
    loss.backward()
    optimizer.step()

    if not args.disable_densification:
        stats = densifier.update(model, render, optimizer, step)
```

这一小段就是 3DGS 训练的核心。下面逐行拆。

### 9.1 随机选一张训练相机

```python
camera_index = random.choice(camera_indices)
camera = train_cameras[camera_index]
```

意思是：

```text
每一步不渲染所有训练图
而是随机选一张图
用这一张图的误差更新整个 Gaussian 场景
```

这类似随机梯度下降。

### 9.2 清空上一轮梯度

```python
optimizer.zero_grad(set_to_none=True)
```

PyTorch 参数会保存 `.grad`。每一轮都要清掉，否则梯度会累加错。

### 9.3 更新 means 的学习率

```python
set_group_lr(optimizer, "means", position_lr(step))
```

位置 `means` 用指数衰减：

文件：`modules/optim.py`

```python
position_lr = exponential_lr(1.6e-4, 1.6e-6, max_steps=args.iterations)
```

直观理解：

```text
训练前期位置可以动大一点
训练后期位置动小一点，做精修
```

### 9.4 渲染当前图

```python
render = renderer.render(model, camera)
```

进入：

文件：`modules/renderer.py`

```python
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
)
```

这一行里面发生了 3DGS 最核心的计算：

```text
3D Gaussian
  -> 相机坐标
  -> 2D 椭圆 splat
  -> 每个像素 alpha 混合
  -> 得到渲染图
```

下面继续拆。

## 10. 第八步计算：3D 点投影到 2D 像素

对第 `i` 个 Gaussian 中心：

```text
mu_i = (x, y, z) in world coordinates
```

先转到相机坐标：

```text
[X_c, Y_c, Z_c, 1]^T = viewmat * [x, y, z, 1]^T
```

然后透视投影：

```text
u_i = fx * X_c / Z_c + cx
v_i = fy * Y_c / Z_c + cy
```

这里：

```text
(u_i, v_i) 是 Gaussian 中心投到图像上的像素位置
Z_c 是深度
```

如果 `Z_c <= near` 或者 `Z_c >= far`，这个 Gaussian 会被裁剪掉。

手算例子：

```text
fx = 100, fy = 100, cx = 50, cy = 50
viewmat = identity
mu = (0.1, 0.0, 2.0)
```

相机坐标就是：

```text
X_c = 0.1
Y_c = 0.0
Z_c = 2.0
```

投影：

```text
u = 100 * 0.1 / 2.0 + 50 = 55
v = 100 * 0.0 / 2.0 + 50 = 50
```

所以这个 Gaussian 中心落在像素附近：

```text
(u, v) = (55, 50)
```

## 11. 第九步计算：3D 椭球投影成 2D 椭圆

Gaussian 不是只投影一个点。它有空间尺度，所以投到屏幕上是一个 2D 椭圆。

3D 协方差：

```text
Sigma_3D = R * diag(sx^2, sy^2, sz^2) * R^T
```

透视投影在当前点附近的雅可比可以记成 `J`。世界到相机旋转记成 `W`。

近似投影后的 2D 协方差：

```text
Sigma_2D = J * W * Sigma_3D * W^T * J^T
```

`gsplat.rasterization()` 会在 CUDA 里做这个计算，还会加上数值稳定项，例如 `eps2d`。

仓库里传入：

文件：`modules/renderer.py`

```python
eps2d=self.eps2d
radius_clip=self.radius_clip
```

最终你可以把每个 Gaussian 理解成屏幕上的一个椭圆：

```text
中心：     (u_i, v_i)
形状：     Sigma_2D_i
深度：     Z_i
颜色：     color_i
透明度：   opacity_i
```

## 12. 第十步计算：球谐颜色怎么来

3DGS 的颜色可以随视角变化。

对第 `i` 个 Gaussian，从相机看它的方向记成：

```text
d_i = normalize(camera_center - mu_i)
```

球谐颜色一般是：

```text
color_i(d_i) = sum over k:
    SH_coeff_i,k * Y_k(d_i)
```

其中：

```text
Y_k(d_i) 是球谐基函数
SH_coeff_i,k 是 learnable 颜色系数
```

在本仓库里：

```text
features_dc + features_rest
  -> 作为 colors 传给 gsplat.rasterization()
  -> gsplat 根据 sh_degree 和视角方向算颜色
```

如果只看 0 阶，颜色退化为：

```text
rgb = SH_C0 * features_dc + 0.5
```

所以初始化时，颜色基本来自 COLMAP 点云 RGB。

训练后，高阶 SH 会学习视角相关效果。

## 13. 第十一步计算：一个像素怎么由多个 Gaussian 合成

现在看某个像素 `p = (u, v)`。

第 `i` 个 Gaussian 对这个像素的覆盖权重：

```text
w_i(p) = exp(-0.5 * (p - m_i)^T * Sigma_2D_i^-1 * (p - m_i))
```

这里：

```text
m_i = (u_i, v_i)
```

把覆盖权重和透明度合成：

```text
a_i(p) = opacity_i * w_i(p)
```

`a_i(p)` 就是这个 Gaussian 在该像素贡献的 alpha。

多个 Gaussian 按深度从前到后合成。前面的会挡住后面的。

设第 `i` 个 Gaussian 之前的透过率是：

```text
T_i(p) = product over j before i:
    (1 - a_j(p))
```

像素颜色：

```text
C(p) = sum over i:
    T_i(p) * a_i(p) * color_i
```

如果背景是白色 `bg = (1, 1, 1)`，最后还要加：

```text
C_final(p) = C(p) + T_final(p) * bg
```

其中：

```text
T_final(p) = product over all i:
    (1 - a_i(p))
```

仓库里白背景来自：

文件：`methods/3dgs/train_3dgs_scene.py`

```python
renderer = GaussianRenderer(background=(1.0, 1.0, 1.0))
```

文件：`modules/renderer.py`

```python
image = image + background.view(3, 1, 1) * (1.0 - alpha)
```

这就是 alpha blending。

## 14. 手算一个单 Gaussian 单像素例子

为了把上面的计算真正走一遍，先忽略 2D 协方差的复杂形状，假设某个 Gaussian 正好落在这个像素中心。

给定：

```text
RGB = (128, 64, 32)
rgb01 = (0.50196, 0.25098, 0.12549)
opacity = 0.1
background = (1, 1, 1)
```

这个 Gaussian 在像素中心，所以：

```text
w(p) = 1
a(p) = opacity * w(p) = 0.1
```

只有一个 Gaussian，没有前面的遮挡，所以：

```text
T = 1
```

像素颜色：

```text
C = T * a * rgb + (1 - a) * background
  = 1 * 0.1 * rgb + 0.9 * (1, 1, 1)
```

逐通道算：

```text
R = 0.1 * 0.50196 + 0.9 = 0.95020
G = 0.1 * 0.25098 + 0.9 = 0.92510
B = 0.1 * 0.12549 + 0.9 = 0.91255
```

所以渲染像素是：

```text
C_render = (0.95020, 0.92510, 0.91255)
```

如果真实 GT 像素是：

```text
C_gt = (0.9, 0.8, 0.7)
```

L1 loss：

```text
L1 = mean(abs(C_render - C_gt))
   = (|0.95020 - 0.9| + |0.92510 - 0.8| + |0.91255 - 0.7|) / 3
   = 0.12928
```

真实训练不是只算一个像素，而是对整张图所有像素求平均。

## 15. 第十二步计算：渲染输出是什么

`GaussianRenderer.render()` 返回：

文件：`modules/renderer.py`

```python
RenderOutput(
    image,
    alpha,
    depth,
    radii,
    means2d,
    metadata,
)
```

每个字段含义：

```text
image
  shape (3, H, W)
  当前相机下的渲染 RGB 图

alpha
  shape (1, H, W)
  每个像素累计不透明度

depth
  可选
  如果 render_mode 带深度，会返回深度图

radii
  Gaussian 在屏幕上的半径信息
  densification 可能用

means2d
  Gaussian 投影到屏幕上的 2D 位置
  反传后它的梯度用于 densification

metadata
  gsplat 返回的额外信息
  当前最关键的是 gaussian_ids
```

训练 loss 只直接用：

```text
render.image
```

densification 会用：

```text
render.means2d
render.radii
render.metadata["gaussian_ids"]
```

## 16. 第十三步计算：图像 loss

训练脚本调用：

文件：`methods/3dgs/train_3dgs_scene.py`

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
def photometric_loss(pred, gt, lambda_dssim=0.2):
    l1 = l1_loss(pred, gt)
    ssim_value = ssim(pred, gt)
    dssim = (1.0 - ssim_value) * 0.5
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim
    return total, {"l1": l1, "ssim": ssim_value, "dssim": dssim, "total": total}
```

公式：

```text
loss = (1 - lambda) * L1 + lambda * DSSIM
DSSIM = (1 - SSIM) / 2
```

默认：

```text
lambda = 0.2
```

所以：

```text
loss = 0.8 * L1 + 0.2 * (1 - SSIM) / 2
```

L1 是逐像素逐通道平均：

```text
L1 = mean over pixels and channels:
    abs(render_image - gt_image)
```

SSIM 是结构相似度，用来约束局部结构，不只是逐像素颜色。

## 17. 第十四步计算：反向传播更新哪些参数

训练脚本：

```python
loss.backward()
optimizer.step()
```

反向传播路径是：

```text
loss
  -> render.image
  -> alpha blending
  -> 每个 Gaussian 的颜色 / opacity / 2D 投影 / 2D 协方差
  -> 3D means / log_scales / quats / logit_opacities / SH features
```

会被更新的参数：

```text
means
features_dc
features_rest
logit_opacities
log_scales
quats
```

优化器由这里创建：

文件：`modules/optim.py`

```python
build_3dgs_optimizer(model, OptimConfig())
```

参数组：

```text
means             lr = 1.6e-4，训练中还会指数衰减
features_dc       lr = 2.5e-3
features_rest     lr = 2.5e-3 / 20
logit_opacities   lr = 5.0e-2
log_scales        lr = 5.0e-3
quats             lr = 1.0e-3
```

Adam 的简化计算：

```text
g_t = d loss / d theta
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

其中 `theta` 可以是：

```text
某个 Gaussian 的位置
某个 Gaussian 的 opacity logit
某个 Gaussian 的 scale
某个 SH 颜色系数
```

## 18. 第十五步计算：为什么需要 densification

如果只优化初始 COLMAP 点，问题是：

```text
COLMAP 点云太稀
有些区域没有 Gaussian
有些边缘细节不够
有些透明度很低的 Gaussian 没贡献
```

所以 3DGS 会动态改变 Gaussian 数量：

```text
clone   复制小 Gaussian，增加局部表达能力
split   把大 Gaussian 分裂成多个小 Gaussian
prune   删除没贡献的 Gaussian
```

训练脚本调用：

文件：`methods/3dgs/train_3dgs_scene.py`

```python
stats = densifier.update(model, render, optimizer, step)
```

真正实现：

文件：`modules/densification.py`

```python
DensificationController.update()
```

## 19. 第十六步计算：densification 用什么依据

关键依据是屏幕空间梯度：

```text
grad_i = norm(d loss / d means2d_i)
```

直观理解：

```text
如果一个 Gaussian 的屏幕位置稍微移动，loss 变化很大
说明这个区域还没拟合好
这个 Gaussian 附近需要更多表达能力
```

仓库里：

文件：`modules/gaussian_model.py`

```python
def accumulate_gradient_stats(self, means2d, visibility=None, use_absgrad=True, indices=None):
    grad = means2d.absgrad 或 means2d.grad
    norms = grad[..., :2].norm(dim=-1)
    gradient_accum += norms
    gradient_count += 1
```

因为当前 renderer 使用 `packed=True`，`means2d` 不是完整 `N` 个 Gaussian 的 dense 数组，所以要靠：

```text
render.metadata["gaussian_ids"]
```

把可见 Gaussian 的梯度加回对应的全局 Gaussian id。

平均梯度：

```text
avg_grad_i = gradient_accum_i / max(gradient_count_i, 1)
```

高梯度判断：

```text
high_grad_i = avg_grad_i >= densify_grad_threshold
```

训练脚本默认传：

```text
--densify-grad-threshold 2e-5
```

## 20. 第十七步计算：clone / split / prune 规则

文件：`modules/densification.py`

```python
max_scale = model.scales.detach().max(dim=-1).values
dense_threshold = scene_extent * percent_dense
clone_mask = high_grad & (max_scale <= dense_threshold)
split_mask = high_grad & (max_scale > dense_threshold)
```

### 20.1 clone

条件：

```text
梯度高
Gaussian 本身比较小
```

公式：

```text
clone_mask_i = high_grad_i and max_scale_i <= scene_extent * percent_dense
```

操作：

```text
复制这个 Gaussian 的所有参数
新 Gaussian 和旧 Gaussian 初始完全一样
后续训练会把它们优化到不同位置或颜色
```

对应函数：

文件：`modules/gaussian_model.py`

```python
def clone(self, mask):
    tensors = selected old tensors
    append_tensors(tensors)
```

### 20.2 split

条件：

```text
梯度高
Gaussian 本身比较大
```

公式：

```text
split_mask_i = high_grad_i and max_scale_i > scene_extent * percent_dense
```

操作：

```text
从原 Gaussian 椭球里采样偏移
生成多个子 Gaussian
子 Gaussian scale 缩小
删除原来的大 Gaussian
```

对应函数：

文件：`modules/gaussian_model.py`

```python
def split(self, mask, num_splits=2, scale_shrink=1.6):
    samples = randn(...)
    local_offsets = samples * scales
    offsets = rotations @ local_offsets
    new_means = old_means + offsets
    new_log_scales = old_log_scales - log(scale_shrink)
```

### 20.3 prune

条件：

```text
opacity 太低
```

公式：

```text
prune_mask_i = opacity_i < min_opacity
```

训练脚本里：

```text
min_opacity = 0.005
```

操作：

```text
删掉几乎透明、对图像没贡献的 Gaussian
```

### 20.4 optimizer state 也要同步

Gaussian 数量变了，Adam 里的动量状态也必须变。

所以 `modules/densification.py` 里还有：

```text
_patch_optimizer_append()
_patch_optimizer_append_and_prune()
_patch_optimizer_prune()
```

否则模型参数数量和 optimizer state 数量对不上，训练会坏。

## 21. 第十八步计算：opacity reset

训练中还会定期 reset opacity：

文件：`modules/densification.py`

```python
if step % opacity_reset_interval == 0:
    model.reset_opacities(reset_opacity)
```

当前配置：

```text
opacity_reset_interval = 3000
reset_opacity = 0.01
```

意思是每 3000 步把透明度压回较低值。

目的：

```text
避免太多 Gaussian 过早变得太不透明
让后续优化和致密化更稳定
```

## 22. 第十九步计算：训练中评估 PSNR

训练时会定期算测试集 PSNR：

文件：`methods/3dgs/train_3dgs_scene.py`

```python
eval_psnr = evaluate_psnr(model, renderer, test_cameras)
```

函数：

```python
@torch.no_grad()
def evaluate_psnr(model, renderer, cameras):
    for camera in cameras:
        render = renderer.render(model, camera)
        psnr = compute_psnr(render.image, camera.image)
    return mean(psnr)
```

PSNR 公式：

```text
MSE = mean((pred - target)^2)
PSNR = -10 * log10(MSE)
```

仓库实现：

文件：`utils/image_utils.py`

```python
def compute_psnr(pred, target):
    mse = mean((pred - target)^2)
    return -10.0 * log10(max(mse, 1e-12))
```

注意：

```text
evaluate_psnr 不反传
只评估当前模型好不好
```

## 23. 第二十步计算：保存 final.ply

训练过程中定期保存：

```text
outputs/3dgs_scene/checkpoints/step_XXXXXX.ply
```

训练结束保存：

```text
outputs/3dgs_scene/final.ply
```

训练脚本：

文件：`methods/3dgs/train_3dgs_scene.py`

```python
save_checkpoint(out_dir / "final.ply", model)
```

内部：

```python
data = gaussians_to_ply_dict(
    model.means,
    model.log_scales,
    model.quats,
    model.logit_opacities,
    model.features_dc,
    model.features_rest,
)
write_ply(path, data)
```

PLY 里存的是 raw 参数：

```text
x, y, z
f_dc_*
f_rest_*
opacity
scale_*
rot_*
```

注意：

```text
这里的 opacity 存的是 logit_opacity
这里的 scale 存的是 log_scale
这里的 rot 存的是 quaternion
```

所以这个 PLY 不是普通 mesh，不是三角面片模型。它是 3DGS 参数文件。

## 24. 第二十一步计算：训练后怎么渲染数据集视角

入口：

```text
methods/3dgs/render_3dgs_views.py
```

运行：

```bash
python methods/3dgs/render_3dgs_views.py \
  --checkpoint outputs/3dgs_scene/final.ply \
  --out outputs/3dgs_scene/renders_test \
  --split test \
  --factor 4
```

计算流程：

```text
read_ply(final.ply)
  -> ply_dict_to_gaussians()
  -> GaussianModel.replace_tensors()
  -> load_colmap_dataset(split="test")
  -> Camera.from_scene_data()
  -> renderer.render()
  -> compute_psnr(render, gt)
  -> save_image(render)
  -> save_image(gt)
```

输出：

```text
*_render.png   模型渲染图
*_gt.png       真实图
平均 PSNR      数值评价
```

这一步和训练时的 render 是同一套计算，只是不再更新参数。

## 25. 第二十二步计算：训练后怎么渲染新视角

入口：

```text
methods/3dgs/render_3dgs_path.py
```

运行插值视角：

```bash
python methods/3dgs/render_3dgs_path.py \
  --checkpoint outputs/3dgs_scene/final.ply \
  --out outputs/3dgs_scene/path_interpolate \
  --mode interpolate \
  --split train \
  --factor 4 \
  --frames 60
```

插值路径的计算：

```text
取 start 相机 c2w_a
取 end 相机 c2w_b
平移线性插值：
  t = (1 - alpha) * t_a + alpha * t_b

旋转球面插值：
  R_a -> quat_a
  R_b -> quat_b
  quat = slerp(quat_a, quat_b, alpha)
  quat -> R

组成新的 c2w
  -> make_camera()
  -> renderer.render()
```

环绕路径：

```bash
python methods/3dgs/render_3dgs_path.py \
  --checkpoint outputs/3dgs_scene/final.ply \
  --out outputs/3dgs_scene/path_orbit \
  --mode orbit \
  --split train \
  --factor 4 \
  --frames 60
```

环绕路径的计算：

```text
用所有训练相机中心求平均 center
按圆周生成新相机位置 position(theta)
用 look_at_c2w(position, center, up) 让相机看向中心
  -> make_camera()
  -> renderer.render()
```

## 26. 把整条计算链合成一句话

完整重建就是：

```text
读 COLMAP:
  qvec/tvec/K/images/points3D

转训练数据:
  c2w, viewmat, K, GT image, sparse point cloud

初始化 Gaussian:
  mu = point_xyz
  scale = mean neighbor distance
  quat = identity
  opacity = 0.1
  SH color = rgb_to_sh(point_rgb)

每步训练:
  随机取一张 Camera
  激活 Gaussian 参数
  3D Gaussian 投影到 2D
  2D Gaussian 按深度 alpha blending
  得到 render.image
  和 GT image 算 L1 + DSSIM
  loss.backward()
  Adam 更新 Gaussian 参数
  用屏幕梯度做 clone/split/prune

训练结束:
  保存 final.ply
  用 final.ply 和任意 Camera 渲染图片
```

## 27. 你跟着代码理解时的阅读顺序

不要一开始就看 `gsplat` CUDA 内部。先按这个顺序看：

```text
1. utils/dataset_loaders.py
   看 load_colmap_dataset() 怎么产出 SceneData

2. modules/camera.py
   看 Camera.from_scene_data(), viewmat, K

3. modules/gaussian_model.py
   看 GaussianModel.from_point_cloud(), activated_tensors()

4. modules/renderer.py
   看 GaussianRenderer.render() 传给 gsplat 什么参数

5. modules/losses.py
   看 photometric_loss()

6. modules/optim.py
   看 build_3dgs_optimizer()

7. modules/densification.py
   看 update(), _densify(), clone/split/prune 条件

8. methods/3dgs/train_3dgs_scene.py
   回来看 main() 怎么把这些模块串起来
```

## 28. 最小手算路线

如果你想真正手算一遍，不要从完整 Curasao 开始。先用这个简化版：

```text
1 个相机
1 个 Gaussian
1 个像素
白背景
只看 0 阶 SH
忽略复杂 2D 协方差，假设 Gaussian 正好覆盖像素中心
```

手算顺序：

```text
1. 给 mu = (0.1, 0, 2)
2. 给 K = [[100,0,50],[0,100,50],[0,0,1]]
3. 算投影 u = 55, v = 50
4. 给 RGB = (128,64,32)，转 rgb01
5. 算 features_dc = (rgb01 - 0.5) / SH_C0
6. 给 opacity = 0.1
7. 假设像素正好在中心，w = 1
8. 算 a = opacity * w = 0.1
9. 白背景合成 C = 0.1 * rgb + 0.9 * white
10. 给 GT = (0.9,0.8,0.7)
11. 算 L1 = 0.12928
12. 理解 loss.backward() 会让 mu/opacity/color/scale 往降低 loss 的方向变
```

这就是 3DGS 的最小计算闭环。完整训练只是把它扩展到：

```text
很多 Gaussian
很多像素
很多相机
复杂 2D 椭圆投影
球谐视角相关颜色
自动微分
动态 clone/split/prune
```
