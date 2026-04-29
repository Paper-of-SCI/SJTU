# utils/

框架无关的数据 I/O 与数学工具包。

**所有模块只依赖 NumPy、Pillow、scipy 和 Python 标准库，无 PyTorch/JAX 依赖。**  
可被 `modules/`（PyTorch 3DGS）和 `methods/seathru_NeRF/`（JAX）同时复用。

---

## 模块一览

| 文件 | 功能 | 主要类 / 函数 |
|------|------|--------------|
| `camera_utils.py` | 相机数学工具 | `qvec_to_rotmat`, `rotmat_to_qvec`, `build_projection_matrix`, `colmap_to_opengl_c2w`, `focal_to_fov`, `fov_to_focal`, `w2c_to_c2w`, `interpolate_camera_path` |
| `colmap_reader.py` | COLMAP 二进制/文本解析 | `COLMAPCamera`, `COLMAPImage`, `COLMAPPoint3D`, `read_colmap_model`, `write_colmap_model` |
| `ply_io.py` | PLY 点云读写 | `read_ply`, `write_ply`, `gaussians_to_ply_dict`, `ply_dict_to_gaussians` |
| `image_utils.py` | 图像 I/O 与质量度量 | `load_image`, `save_image`, `compute_psnr`, `compute_ssim`, `compute_lpips`, `images_to_video`, `linear_to_srgb`, `srgb_to_linear` |
| `dataset_loaders.py` | 多格式数据集加载器 | `SceneData`, `load_colmap_dataset`, `load_llff_dataset`, `load_blender_dataset`, `compute_scene_scale` |

---

## 各模块详细说明

### `camera_utils.py` — 相机数学工具

纯数学运算，无 I/O。

```python
from utils.camera_utils import qvec_to_rotmat, colmap_to_opengl_c2w, build_projection_matrix

# COLMAP 四元数 → 旋转矩阵
R = qvec_to_rotmat(np.array([w, x, y, z]))  # (3, 3)

# COLMAP w2c → OpenGL c2w（含坐标系翻转）
c2w = colmap_to_opengl_c2w(qvec, tvec)  # (4, 4)

# OpenGL 投影矩阵
P = build_projection_matrix(fx, fy, cx, cy, width, height)  # (4, 4)
```

**坐标系约定：**
- COLMAP：`[right, down, forward]`
- OpenGL / NeRF / 3DGS：`[right, up, backward]`
- `colmap_to_opengl_c2w` 自动完成 w2c → c2w 求逆 + 坐标轴翻转。

---

### `colmap_reader.py` — COLMAP 文件解析器

低级二进制/文本解析，返回纯 Python 数据类，不依赖 pycolmap 库。

```python
from utils.colmap_reader import read_colmap_model

# 自动检测 binary 或 text 格式
cameras, images, points3D = read_colmap_model("sparse/0/")

cam = cameras[1]
print(cam.fx, cam.fy, cam.cx, cam.cy)   # 相机内参

img = images[1]
print(img.name, img.qvec, img.tvec)     # 图像文件名、位姿
```

**数据结构：**
- `COLMAPCamera`：`camera_id, model, width, height, params`，提供 `fx/fy/cx/cy` 属性
- `COLMAPImage`：`image_id, qvec(4), tvec(3), camera_id, name, xys(M,2), point3D_ids(M)`
- `COLMAPPoint3D`：`point3D_id, xyz(3), rgb(3), error, image_ids, point2D_idxs`

---

### `ply_io.py` — PLY 点云读写

支持 ASCII 和 binary_little_endian 格式，内置 3DGS checkpoint 的属性命名规范。

```python
from utils.ply_io import read_ply, write_ply, gaussians_to_ply_dict, ply_dict_to_gaussians

# 读取 PLY
data = read_ply("point_cloud.ply")   # Dict[str, np.ndarray]
print(data.keys())  # 'x','y','z','f_dc_0','f_rest_0','opacity','scale_0','rot_0' ...

# 将高斯参数转为 PLY 并保存
ply_dict = gaussians_to_ply_dict(means, scales, rotations, opacities, sh_dc, sh_rest)
write_ply("output.ply", ply_dict, binary=True)

# 从 PLY 恢复参数
means, scales, rotations, opacities, sh_dc, sh_rest = ply_dict_to_gaussians(data)
```

**3DGS PLY 属性命名规范：**
- 位置：`x, y, z`
- SH DC：`f_dc_0, f_dc_1, f_dc_2`
- SH 高阶：`f_rest_0` … `f_rest_N`
- 不透明度：`opacity`（pre-sigmoid）
- 尺度：`scale_0, scale_1, scale_2`（log 空间）
- 旋转：`rot_0, rot_1, rot_2, rot_3`（wxyz 四元数）

---

### `image_utils.py` — 图像工具

无 PyTorch 依赖，SSIM 手动实现（scipy.ndimage），与 3DGS 论文评估协议一致。

```python
from utils.image_utils import load_image, save_image, compute_psnr, compute_ssim

img = load_image("photo.jpg", as_float=True)  # (H, W, 3) float32 [0,1]
save_image("output.png", img)

psnr = compute_psnr(pred, gt)   # float，单位 dB
ssim = compute_ssim(pred, gt)   # float，[0,1]
```

---

### `dataset_loaders.py` — 数据集加载器

统一接口，三种格式均返回 `SceneData`。

```python
from utils.dataset_loaders import load_colmap_dataset, load_llff_dataset, load_blender_dataset

# COLMAP 格式（SeaThru 数据集）
scene = load_colmap_dataset(
    "src/datasets/SeathruNeRF_dataset/Panama",
    split="train",
    load_images=True,
)
print(scene.c2w_matrices.shape)      # (N, 4, 4)
print(scene.point_cloud_xyz.shape)   # (M, 3)

# LLFF 格式
scene = load_llff_dataset("data/llff_scene", split="train")

# Blender 合成格式
scene = load_blender_dataset("data/lego", split="train", white_background=True)
```

**`SceneData` 字段：**
```python
image_paths: List[str]           # 绝对路径
images: Optional[np.ndarray]     # (N, H, W, 3) float32 [0,1]
c2w_matrices: np.ndarray         # (N, 4, 4) float64，OpenGL 坐标系
fx, fy, cx, cy: np.ndarray       # 内参，(N,) 或标量
width, height: int
point_cloud_xyz: Optional[np.ndarray]   # (M, 3)
point_cloud_rgb: Optional[np.ndarray]   # (M, 3) uint8
near, far, scene_scale: float
```

---

## 快速上手

```python
from utils import load_colmap_dataset, compute_psnr, read_colmap_model

# 加载 SeaThru 数据集（自动识别 images_wb/ 目录）
scene = load_colmap_dataset(
    "src/datasets/SeathruNeRF_dataset/Panama",
    split="train",
)
print(f"共 {len(scene.image_paths)} 张训练图像")
print(f"点云点数：{len(scene.point_cloud_xyz)}")
print(f"场景尺度：{scene.scene_scale:.3f}")
```

---

## 依赖

```
numpy
Pillow
scipy
opencv-python     # 仅 images_to_video 需要
lpips             # 仅 compute_lpips 需要（可选）
```
