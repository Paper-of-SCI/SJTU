import torch
from torch import nn
from methods.leo_3DGS.contracts.gaussian import GaussianParameters
from methods.leo_3DGS.functions.initialGS import main_initialize_gaussians_from_point_cloud
from methods.leo_3DGS.utils_.loadData import load_colmap_cameras_bin, load_colmap_images_bin
from PIL import Image
import math
import sys
from pathlib import Path
import numpy as np


class GaussianModel(nn.Module):
    def __init__(self, params: GaussianParameters):
        super().__init__()

        self.means = nn.Parameter(params.means)
        self.features_dc = nn.Parameter(params.features_dc)
        self.features_rest = nn.Parameter(params.features_rest)
        self.opacity_logits = nn.Parameter(params.opacity_logits)
        self.rotations = nn.Parameter(params.rotations)
        self.log_scales = nn.Parameter(params.log_scales)

    '''
     @property 是让访问方法像访问属性一样，简化了代码的使用方式，使得代码更清晰易读。
    正常方法调用应该是:model.opacities()
    但加了 @property 后,可以这样访问:model.opacities
    '''
    
  
    @property
    def opacities(self):
        return torch.sigmoid(self.opacity_logits)

    @property
    def scales(self):
        return torch.exp(self.log_scales)
    
    # 将a0和其他a1-an 拼到一起，得到完整的SH系数
    @property
    def features(self):
        return torch.cat([self.features_dc, self.features_rest], dim=1)
    
    # 把每个 Gaussian 的四元数旋转重新归一化成单位四元数：四元数要表示合法旋转，长度必须是 1
    # \(\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2} = 1\)
    @property
    def normalized_rotations(self):
        return torch.nn.functional.normalize(self.rotations, dim=1)
    
    def colors_from_sh(self, dirs: torch.Tensor, degree: int = 3) -> torch.Tensor:
        """
        根据观察方向 dirs 计算每个 Gaussian 的 RGB 颜色。

        Args:
            dirs: shape [N, 3]，每个 Gaussian 对应一个单位观察方向。
            degree: 使用的 SH 阶数，支持 0 到 3。

        Returns:
            colors: shape [N, 3]，范围 [0, +inf]。
        """
        if degree < 0 or degree > 3:
            raise ValueError("degree must be between 0 and 3")

        features = self.features
        required_coeffs = (degree + 1) ** 2
        if features.shape[1] < required_coeffs:
            raise ValueError(
                f"features has {features.shape[1]} SH coeffs, but degree={degree} requires {required_coeffs}"
            )
        dirs = torch.nn.functional.normalize(dirs, dim=1)

        x = dirs[:, 0:1]
        y = dirs[:, 1:2]
        z = dirs[:, 2:3]

        pi = torch.tensor(torch.pi, dtype=features.dtype, device=features.device)

        c0 = 1.0 / (2.0 * torch.sqrt(pi))
        c1 = torch.sqrt(torch.tensor(3.0, dtype=features.dtype, device=features.device) / (4.0 * pi))

        c2_0 = torch.sqrt(torch.tensor(15.0, dtype=features.dtype, device=features.device) / (4.0 * pi))
        c2_1 = -torch.sqrt(torch.tensor(15.0, dtype=features.dtype, device=features.device) / (4.0 * pi))
        c2_2 = torch.sqrt(torch.tensor(5.0, dtype=features.dtype, device=features.device) / (16.0 * pi))
        c2_3 = -torch.sqrt(torch.tensor(15.0, dtype=features.dtype, device=features.device) / (4.0 * pi))
        c2_4 = torch.sqrt(torch.tensor(15.0, dtype=features.dtype, device=features.device) / (16.0 * pi))

        c3_0 = -torch.sqrt(torch.tensor(35.0, dtype=features.dtype, device=features.device) / (32.0 * pi))
        c3_1 = torch.sqrt(torch.tensor(105.0, dtype=features.dtype, device=features.device) / (4.0 * pi))
        c3_2 = -torch.sqrt(torch.tensor(21.0, dtype=features.dtype, device=features.device) / (32.0 * pi))
        c3_3 = torch.sqrt(torch.tensor(7.0, dtype=features.dtype, device=features.device) / (16.0 * pi))
        c3_4 = -torch.sqrt(torch.tensor(21.0, dtype=features.dtype, device=features.device) / (32.0 * pi))
        c3_5 = torch.sqrt(torch.tensor(105.0, dtype=features.dtype, device=features.device) / (16.0 * pi))
        c3_6 = -torch.sqrt(torch.tensor(35.0, dtype=features.dtype, device=features.device) / (32.0 * pi))

        result = c0 * features[:, 0, :]

        if degree >= 1:
            result = (
                result
                - c1 * y * features[:, 1, :]
                + c1 * z * features[:, 2, :]
                - c1 * x * features[:, 3, :]
            )

        if degree >= 2:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z

            result = (
                result
                + c2_0 * xy * features[:, 4, :]
                + c2_1 * yz * features[:, 5, :]
                + c2_2 * (2.0 * zz - xx - yy) * features[:, 6, :]
                + c2_3 * xz * features[:, 7, :]
                + c2_4 * (xx - yy) * features[:, 8, :]
            )

        if degree >= 3:
            result = (
                result
                + c3_0 * y * (3.0 * x * x - y * y) * features[:, 9, :]
                + c3_1 * x * y * z * features[:, 10, :]
                + c3_2 * y * (4.0 * z * z - x * x - y * y) * features[:, 11, :]
                + c3_3 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y) * features[:, 12, :]
                + c3_4 * x * (4.0 * z * z - x * x - y * y) * features[:, 13, :]
                + c3_5 * z * (x * x - y * y) * features[:, 14, :]
                + c3_6 * x * (x * x - 3.0 * y * y) * features[:, 15, :]
            )

        colors = result + 0.5
        return torch.clamp_min(colors, 0.0)
        
def world_covariances_to_camera(
    cov3d: torch.Tensor,
    image,
) -> torch.Tensor:
    """
    Args:
        cov3d: [N, 3, 3]

    Returns:
        cov_cam: [N, 3, 3]
    """
    device = cov3d.device

    qvec = torch.from_numpy(image.qvec).float().to(device)
    rotation_cam = qvec_to_rotmat(qvec)  # [3, 3]

    cov_cam = rotation_cam[None, :, :] @ cov3d @ rotation_cam.T[None, :, :]
    return cov_cam
    
def build_3d_covariances(model: GaussianModel) -> torch.Tensor:
    r"""
    根据 Gaussian 的 scale 和 rotation 构造世界坐标系下的 3D covariance。

    每个 Gaussian 的 3D covariance 为：

    \[
    \Sigma_w = R_g S S^\top R_g^\top
    \]

    其中：

    - \(R_g\)：Gaussian 自己的旋转矩阵
    - \(S = \mathrm{diag}(s_x, s_y, s_z)\)：Gaussian 三轴尺度矩阵

    Args:
        model: GaussianModel。

    Returns:
        cov3d: shape [N, 3, 3]。
    """
    rotations = model.normalized_rotations      # [N, 4]
    scales = model.scales                       # [N, 3]

    rotation_mats = quaternion_to_rotmat_batch(rotations)  # [N, 3, 3]
    scale_mats = torch.diag_embed(scales)                  # [N, 3, 3]

    transform = rotation_mats @ scale_mats
    cov3d = transform @ transform.transpose(1, 2)

    return cov3d
    
def quaternion_to_rotmat_batch(q: torch.Tensor) -> torch.Tensor:
    """
    将一批四元数转成旋转矩阵。

    Args:
        q: shape [N, 4]，顺序为 [w, x, y, z]。

    Returns:
        rotation_mats: shape [N, 3, 3]。
    """
    q = torch.nn.functional.normalize(q, dim=1)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    row0 = torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
    ], dim=1)

    row1 = torch.stack([
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
    ], dim=1)

    row2 = torch.stack([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dim=1)

    return torch.stack([row0, row1, row2], dim=1)    
    
def qvec_to_rotmat(qvec: torch.Tensor) -> torch.Tensor:
    qvec = qvec / torch.linalg.norm(qvec)
    w, x, y, z = qvec

    row0 = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
    ])

    row1 = torch.stack([
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
    ])

    row2 = torch.stack([
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    ])

    return torch.stack([row0, row1, row2], dim=0)

def gaussian_view_dirs(model: GaussianModel, image) -> torch.Tensor:
    """
    计算当前相机视角下，每个 Gaussian 的观察方向。

    方向定义为：从相机中心指向 Gaussian 中心。

    公式:
        C = -R^T t
        d_i = normalize(mu_i - C)

    Args:
        model: GaussianModel，包含 means，shape [N, 3]。
        image: COLMAP image 数据，包含 qvec 和 tvec。

    Returns:
        dirs: shape [N, 3]，每个 Gaussian 的单位观察方向。
    """
    device = model.means.device

    qvec = torch.from_numpy(image.qvec).float().to(device)
    tvec = torch.from_numpy(image.tvec).float().to(device)

    rotation = qvec_to_rotmat(qvec)

    camera_center = -rotation.T @ tvec

    dirs = model.means - camera_center[None, :]
    dirs = torch.nn.functional.normalize(dirs, dim=1)

    return dirs
    
def project_gaussian_means(model, image, camera):
    device = model.means.device
    qvec = torch.from_numpy(image.qvec).float().to(device)
    tvec = torch.from_numpy(image.tvec).float().to(device)
    

    R = qvec_to_rotmat(qvec)

    means = model.means

    xyz_cam = means @ R.T + tvec[None, :]

    x = xyz_cam[:, 0]
    y = xyz_cam[:, 1]
    z = xyz_cam[:, 2]

    u = camera.fx * (x / z) + camera.cx
    v = camera.fy * (y / z) + camera.cy

    pixels = torch.stack([u, v], dim=1)

    valid = (
        (z > 0) &
        (u >= 0) &
        (u < camera.width) &
        (v >= 0) &
        (v < camera.height)
    )

    return pixels, z, valid, xyz_cam

def covariance_2d_to_radius(cov2d: torch.Tensor, sigma_extent: float = 3.0) -> torch.Tensor:
    """
    根据 2D covariance 估计每个 Gaussian 在屏幕上的覆盖半径。

    Args:
        cov2d: shape [N, 2, 2]
        sigma_extent: 覆盖几个标准差，通常取 3。

    Returns:
        radii: shape [N]，每个 Gaussian 的像素半径。
    """
    eigenvalues = torch.linalg.eigvalsh(cov2d)
    max_eigenvalues = eigenvalues[:, -1].clamp_min(1e-8)
    radii = torch.ceil(sigma_extent * torch.sqrt(max_eigenvalues)).long()
    return radii

def project_covariances_to_2d(
    cov_cam: torch.Tensor,
    xyz_cam: torch.Tensor,
    camera,
) -> torch.Tensor:
    r"""
    将相机坐标系下的 3D covariance 投影成屏幕 2D covariance。

    透视投影为：

    \[
    u = f_x \frac{x}{z} + c_x
    \]

    \[
    v = f_y \frac{y}{z} + c_y
    \]

    对 \((x,y,z)\) 的 Jacobian 为：

    \[
    J =
    \begin{bmatrix}
    \frac{f_x}{z} & 0 & -\frac{f_x x}{z^2} \\
    0 & \frac{f_y}{z} & -\frac{f_y y}{z^2}
    \end{bmatrix}
    \]

    2D covariance：

    \[
    \Sigma_{2D} = J \Sigma_c J^\top
    \]

    Args:
        cov_cam: shape [N, 3, 3]，相机坐标系下的 3D covariance。
        xyz_cam: shape [N, 3]，Gaussian 中心的相机坐标。
        camera: 包含 fx, fy 的相机内参。

    Returns:
        cov2d: shape [N, 2, 2]，屏幕空间 2D covariance。
    """
    device = cov_cam.device
    dtype = cov_cam.dtype

    x = xyz_cam[:, 0]
    y = xyz_cam[:, 1]
    z = xyz_cam[:, 2].clamp_min(1e-6)

    fx = torch.tensor(camera.fx, dtype=dtype, device=device)
    fy = torch.tensor(camera.fy, dtype=dtype, device=device)

    zeros = torch.zeros_like(z)

    j00 = fx / z
    j01 = zeros
    j02 = -fx * x / (z * z)

    j10 = zeros
    j11 = fy / z
    j12 = -fy * y / (z * z)

    row0 = torch.stack([j00, j01, j02], dim=1)
    row1 = torch.stack([j10, j11, j12], dim=1)

    jacobian = torch.stack([row0, row1], dim=1)  # [N, 2, 3]

    cov2d = jacobian @ cov_cam @ jacobian.transpose(1, 2)

    # 加一点对角正则，避免后面求逆时数值不稳定。
    eps = torch.tensor(1e-6, dtype=dtype, device=device)
    eye = torch.eye(2, dtype=dtype, device=device)[None, :, :]
    cov2d = cov2d + eps * eye

    return cov2d

def _load_official_rasterizer():
    rasterizer_root = (
        Path(__file__).resolve().parents[1]
        / "adapters"
        / "diff-gaussian-rasterization"
    )

    if str(rasterizer_root) not in sys.path:
        sys.path.insert(0, str(rasterizer_root))

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    return GaussianRasterizationSettings, GaussianRasterizer


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def get_projection_matrix(
    znear: float,
    zfar: float,
    fovx: float,
    fovy: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tan_half_fovy = math.tan(fovy / 2.0)
    tan_half_fovx = math.tan(fovx / 2.0)

    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right

    projection = torch.zeros((4, 4), dtype=dtype, device=device)

    z_sign = 1.0

    projection[0, 0] = 2.0 * znear / (right - left)
    projection[1, 1] = 2.0 * znear / (top - bottom)
    projection[0, 2] = (right + left) / (right - left)
    projection[1, 2] = (top + bottom) / (top - bottom)
    projection[3, 2] = z_sign
    projection[2, 2] = z_sign * zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)

    return projection

def render_original_cuda(
    model: GaussianModel,
    image,
    camera,
    bg_color: torch.Tensor | None = None,
    sh_degree: int = 3,
    scaling_modifier: float = 1.0,
    znear: float = 0.01,
    zfar: float = 100.0,
    debug: bool = False,
) -> dict:
    """
    调用 diff-gaussian-rasterization 的原版 CUDA rasterizer。

    Returns:
        {
            "render": [3, H, W],
            "alpha": [1, H, W],
            "radii": [N],
            "visibility_filter": [N],
            "viewspace_points": [N, 3],
        }
    """
    if not model.means.is_cuda:
        raise ValueError("diff-gaussian-rasterization 需要 CUDA Tensor，请先 model.cuda()")

    if sh_degree < 0 or sh_degree > 3:
        raise ValueError("sh_degree must be between 0 and 3")

    GaussianRasterizationSettings, GaussianRasterizer = _load_official_rasterizer()

    device = model.means.device
    dtype = model.means.dtype

    if bg_color is None:
        bg_color = torch.zeros((3,), dtype=dtype, device=device)
    else:
        bg_color = bg_color.to(device=device, dtype=dtype)

    qvec = torch.from_numpy(image.qvec).to(device=device, dtype=dtype)
    tvec = torch.from_numpy(image.tvec).to(device=device, dtype=dtype)

    rotation_w2c = qvec_to_rotmat(qvec)

    world_to_camera = torch.eye(4, dtype=dtype, device=device)
    world_to_camera[:3, :3] = rotation_w2c
    world_to_camera[:3, 3] = tvec

    # 原版 rasterizer 使用转置后的矩阵布局。
    world_view_transform = world_to_camera.transpose(0, 1).contiguous()

    fovx = focal_to_fov(camera.fx, camera.width)
    fovy = focal_to_fov(camera.fy, camera.height)

    projection_matrix = get_projection_matrix(
        znear=znear,
        zfar=zfar,
        fovx=fovx,
        fovy=fovy,
        device=device,
        dtype=dtype,
    ).transpose(0, 1).contiguous()

    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )

    camera_center = -rotation_w2c.T @ tvec

    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=float(scaling_modifier),
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=int(sh_degree),
        campos=camera_center,
        prefiltered=False,
        debug=bool(debug),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3d = model.means
    means2d = torch.zeros_like(
        means3d,
        dtype=means3d.dtype,
        device=device,
        requires_grad=True,
    )

    try:
        means2d.retain_grad()
    except RuntimeError:
        pass

    rendered_image, rendered_alpha, radii = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=model.features,
        colors_precomp=None,
        opacities=model.opacities,
        scales=model.scales,
        rotations=model.normalized_rotations,
        cov3D_precomp=None,
    )

    return {
        "render": rendered_image,
        "alpha": rendered_alpha,
        "radii": radii,
        "visibility_filter": radii > 0,
        "viewspace_points": means2d,
    }
    
if __name__ == "__main__":
    # 这里可以进行一些简单的测试，例如创建一个 GaussianModel 实例，检查参数的形状等。
    data = main_initialize_gaussians_from_point_cloud("src/datasets/SeathruNeRF_dataset/IUI3-RedSea/undistorted_pinhole/sparse/0/points3D.bin")
    model = GaussianModel(data).cuda()
    

    cameras = load_colmap_cameras_bin(
        "src/datasets/SeathruNeRF_dataset/IUI3-RedSea/undistorted_pinhole/sparse/0/cameras.bin"
    )

    images = load_colmap_images_bin(
        "src/datasets/SeathruNeRF_dataset/IUI3-RedSea/undistorted_pinhole/sparse/0/images.bin"
    )
    
    # 图片按照命名排序
    image = sorted(images.values(), key=lambda x: x.name)[1]
    
    # 获取该图片对应的相机信息
    camera = cameras[image.camera_id]
    
    # 将GS点投影到像平面。pixels是高斯球在像平面坐标，depths就是相机坐标系下gs的3D点的z值，valid是布尔掩码，判断高斯球是否落在像平面里。
    pixels, depths, valid, xyz_cam = project_gaussian_means(model, image, camera)
    
    # 获取每个高斯球的观察方向，dirs是一个[N, 3]的张量，每行是一个高斯球的观察方向单位向量。
    dirs = gaussian_view_dirs(model, image)
    
    # 颜色是根据观察方向和SH系数计算出来的，范围是[0, 1]，有可能超过1，因为训练时候没限制上限。理论是0~1.
    colors = model.colors_from_sh(dirs, degree=3)
   
    # rendered = render_points_debug(model, image, camera, radius=1, sh_degree=3)
    # out = (rendered.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # Image.fromarray(out).save("render_debug.png")
    
  

    # pkg = render_original_cuda(
    #     model=model,
    #     image=image,
    #     camera=camera,
    #     bg_color=torch.zeros(3, device="cuda"),
    #     sh_degree=3,
    # )

    # rendered = pkg["render"]  # [3, H, W]
    # rendered_image = rendered.detach().clamp(0.0, 1.0)
    # rendered_image = rendered_image.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    # rendered_image = (rendered_image.cpu().numpy() * 255).astype(np.uint8)

    # Image.fromarray(rendered_image).save("render_cuda.png")

    
    
    '''
    以下是验证gs球投影到图片上的二维点是否正确的代码。
    '''
    image_path = "src/datasets/SeathruNeRF_dataset/IUI3-RedSea/undistorted_pinhole/images/" + image.name

    gt = Image.open(image_path).convert("RGB")
    canvas = np.array(gt)
    
    points_2d = pixels[valid].detach().cpu().numpy()

    for u, v in points_2d:  # 每 20 个画一个，避免太密
        u = int(round(u))
        v = int(round(v))

        if 0 <= u < camera.width and 0 <= v < camera.height:
            canvas[max(v - 1, 0): min(v + 2, camera.height),
                max(u - 1, 0): min(u + 2, camera.width)] = [255, 0, 0]
    out = Image.fromarray(canvas)
    out.save("projection_debug.png")
