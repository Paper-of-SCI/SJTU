# 这里存放各种数据集
数据集下载到`src/datasets`目录下，下载链接如下：

- [SeathruNeRF_dataset](https://drive.google.com/uc?export=download&id=1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT) 

## SeathruNeRF_dataset 目录说明

以 `SeathruNeRF_dataset/Curasao` 为例：

| 路径/文件 | 数据类型 | 主要内容 | 用途 |
|---|---|---|---|
| `images_wb/` | 图像目录 | 白平衡后的输入图片，格式通常是 `.png` | 训练/测试时作为 GT 图像 |
| `sparse/0/` | COLMAP 稀疏重建目录 | COLMAP 生成的相机、图片位姿、稀疏点云 | 给 3DGS/NeRF 提供相机参数和初始化点云 |
| `poses_bounds.npy` | NumPy 数组文件 | 每张图的相机位姿、图像尺寸、焦距、near/far bounds | LLFF/NeRF 风格代码常用的相机数据格式 |

## sparse/0 文件说明

| 文件 | 数据类型 | 主要内容 | 用途 |
|---|---|---|---|
| `cameras.bin` | COLMAP 二进制文件 | 相机内参，例如宽高、焦距 `fx/fy`、主点 `cx/cy`、相机模型参数 | 构建相机内参矩阵 `K` |
| `images.bin` | COLMAP 二进制文件 | 每张图片的外参：旋转 `qvec`、平移 `tvec`、图片名、2D 特征点与 3D 点关联 | 构建每张图的相机位姿 |
| `points3D.bin` | COLMAP 二进制文件 | 稀疏三维点云：坐标 `xyz`、颜色 `rgb`、重投影误差等 | 3DGS 初始化高斯点 |
| `project.ini` | COLMAP 配置文件 | COLMAP 工程配置 | 一般训练时不直接使用 |

这些 `.bin` 文件不是普通文本，需要用 COLMAP 或项目里的 `utils/colmap_reader.py` 读取。

## poses_bounds.npy 格式

`poses_bounds.npy` 是一个 NumPy 数组。当前 `Curasao/poses_bounds.npy` 的形状是 `(21, 17)`：

- `21`：一共有 21 张图。
- `17`：每张图用 17 个数字描述相机和深度范围。

每一行 `row` 都可以拆成两部分：

| 部分 | 写法 | 含义 |
|---|---|---|
| 相机信息 | `row[:15].reshape(3, 5)` | 一个 `3 x 5` 矩阵，前 4 列是相机位姿，最后 1 列是图像高、宽、焦距 |
| 深度范围 | `row[15:17]` | 两个数，分别是 near 和 far |

把 `row[:15]` 变成 `3 x 5` 后长这样：

```text
[
  [r00, r01, r02, tx, H],
  [r10, r11, r12, ty, W],
  [r20, r21, r22, tz, focal],
]
```

其中：

| 字段 | 含义 |
|---|---|
| `r00 ~ r22` | 相机旋转矩阵 `R` |
| `tx, ty, tz` | 相机平移位置 `t` |
| `H` | 图像高度 |
| `W` | 图像宽度 |
| `focal` | 焦距 |

所以它其实是把下面两个东西压在一起：

```text
相机位姿:
[
  [r00, r01, r02, tx],
  [r10, r11, r12, ty],
  [r20, r21, r22, tz],
]

图像参数:
H, W, focal
```

当前数据里第一张图的例子：

```text
row[:15].reshape(3, 5) =
[
  [-0.0205,  0.9969,  0.0759,  2.4753, 1182.0],
  [ 0.9803,  0.0051,  0.1974,  2.8254, 1776.0],
  [ 0.1964,  0.0784, -0.9774, -2.5449, 1960.08],
]

row[15:17] = [14.3091, 45.7712]
```

对应解释：

| 数据 | 含义 |
|---|---|
| 前 `3 x 4` | 第一张图的相机位姿 |
| `1182.0` | 图像高度 `H` |
| `1776.0` | 图像宽度 `W` |
| `1960.08` | 焦距 `focal` |
| `14.3091` | near |
| `45.7712` | far |

读取代码：

```python
import numpy as np

poses_bounds = np.load("src/datasets/SeathruNeRF_dataset/Curasao/poses_bounds.npy")

row = poses_bounds[0]
pose_hwf = row[:15].reshape(3, 5)
bounds = row[15:17]

pose = pose_hwf[:, :4]   # 3x4 相机位姿
height = pose_hwf[0, 4]
width = pose_hwf[1, 4]
focal = pose_hwf[2, 4]
near, far = bounds
```
