# utils/

这里放不依赖训练流程的纯工具。

## 模块职责

| 模块 | 主要职责 | 核心接口 | 依赖关系 |
|---|---|---|---|
| `camera_utils.py` | 提供纯 NumPy 相机数学工具，包括四元数/旋转矩阵转换、内参矩阵、COLMAP 位姿到 OpenGL `c2w` 转换、场景尺度估计。 | `qvec_to_rotmat`, `rotmat_to_qvec`, `intrinsic_matrix`, `colmap_image_to_c2w`, `estimate_scene_extent` | 依赖 `numpy` |
| `colmap_reader.py` | 读取 COLMAP sparse model，支持 `cameras/images/points3D` 的 `.bin` 和 `.txt`。 | `read_colmap_model`, `ColmapCamera`, `ColmapImage`, `ColmapPoint3D` | 依赖标准库、`numpy` |
| `dataset_loaders.py` | 加载 COLMAP/LLFF 数据并统一成 `SceneData`；只返回数据，不创建模型、不启动训练。 | `SceneData`, `load_colmap_dataset`, `load_llff_dataset` | 依赖 `numpy`、`camera_utils`、`colmap_reader`、`image_utils` |
| `image_utils.py` | 图像读写和基础指标计算。 | `load_image`, `save_image`, `compute_psnr`, `compute_ssim` | 依赖 `numpy`、`Pillow`；`compute_ssim` 额外依赖 `scipy` |
| `ply_io.py` | 3DGS Gaussian checkpoint 的 PLY 字段转换、ASCII PLY 读写。 | `gaussians_to_ply_dict`, `ply_dict_to_gaussians`, `read_ply`, `write_ply` | 依赖标准库、`numpy` |

## 单模块测试方式

| 模块 | 测试命令 | 预期结果 | 备注 |
|---|---|---|---|
| 全部语法 | `python - <<'PY'`<br>`from pathlib import Path`<br>`for p in list(Path('utils').glob('*.py')):`<br>`    compile(p.read_text(), str(p), 'exec')`<br>`print('syntax ok')`<br>`PY` | 输出 `syntax ok` | 不需要 `torch` 或 `gsplat`。 |
| `camera_utils.py` | `python - <<'PY'`<br>`import numpy as np`<br>`from utils.camera_utils import qvec_to_rotmat, intrinsic_matrix, colmap_image_to_c2w`<br>`print(qvec_to_rotmat(np.array([1,0,0,0])).shape)`<br>`print(intrinsic_matrix(500,500,320,240))`<br>`print(colmap_image_to_c2w(np.array([1,0,0,0]), np.zeros(3)).shape)`<br>`PY` | 输出 `(3,3)`、内参矩阵、`(4,4)` | 只依赖 `numpy`。 |
| `colmap_reader.py` | `python - <<'PY'`<br>`from utils.colmap_reader import read_colmap_model`<br>`cams, imgs, pts = read_colmap_model('src/datasets/SeathruNeRF_dataset/Curasao/sparse/0')`<br>`print(len(cams), len(imgs), len(pts))`<br>`PY` | 输出相机、图像、点云数量 | 需要本仓库当前数据集存在。 |
| `dataset_loaders.py` | `python - <<'PY'`<br>`from utils.dataset_loaders import load_colmap_dataset, load_llff_dataset`<br>`scene = load_colmap_dataset('src/datasets/SeathruNeRF_dataset/Curasao', load_images=False)`<br>`print(len(scene.image_paths), scene.c2w_matrices.shape, scene.point_cloud_xyz.shape)`<br>`scene2 = load_llff_dataset('src/datasets/SeathruNeRF_dataset/Curasao', load_images=False)`<br>`print(len(scene2.image_paths), scene2.width, scene2.height)`<br>`PY` | 输出图像数量、位姿 shape、点云 shape、分辨率 | `load_images=False` 可以避免读入所有图片。 |
| `image_utils.py` | `python - <<'PY'`<br>`from utils.image_utils import load_image, compute_psnr`<br>`img = load_image('src/datasets/SeathruNeRF_dataset/Curasao/images_wb/MTN_1288.png')`<br>`print(img.shape, img.dtype, compute_psnr(img, img))`<br>`PY` | 输出图像 shape、`float32`、很高的 PSNR | `compute_ssim` 需要 `scipy`，此命令不测 SSIM。 |
| `ply_io.py` | `python - <<'PY'`<br>`import os, tempfile, numpy as np`<br>`from utils.ply_io import gaussians_to_ply_dict, write_ply, read_ply, ply_dict_to_gaussians`<br>`n = 2`<br>`path = os.path.join(tempfile.gettempdir(), 'sjtu_utils_ply_test.ply')`<br>`data = gaussians_to_ply_dict(np.zeros((n,3)), np.zeros((n,3)), np.tile([[1,0,0,0]], (n,1)), np.zeros((n,1)), np.zeros((n,1,3)), np.zeros((n,15,3)))`<br>`write_ply(path, data)`<br>`print([x.shape for x in ply_dict_to_gaussians(read_ply(path))])`<br>`PY` | 输出 6 个 Gaussian 参数数组的 shape | 使用临时目录，不写入仓库。 |

`modules/` 负责 PyTorch/gsplat 功能模块；`utils/` 只做数据和格式处理。
