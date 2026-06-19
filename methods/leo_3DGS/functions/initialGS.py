from methods.leo_3DGS.utils_.loadData import load_colmap_points3d_bin
import torch
from methods.leo_3DGS.contracts.gaussian import GaussianParameters
import math

'''
Gaussians球有以下参数

means          -> 高斯中心                   [N, 3]
features_dc    -> 初始颜色SH系数             [N, 1, 3]
features_rest  -> 其他颜色SH系数             [N, num_rest_coeffs, 3] num_rest_coeffs = (sh_degree + 1) ** 2 - 1
log_scales     -> 高斯大小的 log 参数         [N, 3]             N表示点云个数即高斯球个数
rotations      -> 四元数旋转                 [N, 4]
opacity_logits -> 不透明度的 logit 参数       [N, 1]

'''




def main_initialize_gaussians_from_point_cloud(path: str):
    point_cloud = load_colmap_points3d_bin(path)
    
    means = torch.from_numpy(point_cloud.xyz).float()
    colors = torch.from_numpy(point_cloud.colors).float()
    
    num_gaussians = means.shape[0]
    
    '''
    初始化SH系数
    '''
    SH_C0 = 1.0 / (2.0 * math.sqrt(math.pi))
    features_dc = (colors - 0.5) / SH_C0
    # 将 features_dc 从 [N, 3] 扩展到 [N, 1, 3]，以便后续与 SH 基函数相乘时广播。
    features_dc = features_dc[:, None, :]
    
    
    '''
    初始化高斯球的opacity_logits为一个较小的值，例如0.1. 
    opacity肯定是0~1之间的，然后为了让后面训练好优化opacity所以，这里将0.1反向sigmod得到一个logit值，作为初始值。
    opacity_logits = initial_opacity_logits(0.1, num_gaussians).
    所以后面训练得出的是opacity_logits。opacity = sigmoid(opacity_logits)。
    '''
    opacity_logits = initial_opacity_logits(0.1, num_gaussians)
    
    '''
    初始化高斯球的rotations为单位四元数。
    '''
    rotations = initial_rotations(num_gaussians)
    
    '''
    按原版 3DGS 的点云最近邻距离初始化 scale。
    原版用每个点到最近 3 个点的平方距离均值 dist2，然后:
        scale = sqrt(dist2)
        log_scale = log(scale)
    '''
    log_scales = initial_log_scales_from_nearest_neighbors(means)

    '''
    这里是3阶SH其他颜色系数
    '''
    sh_degree = 3
    num_rest_coeffs = (sh_degree + 1) ** 2 - 1 # SH degree 3 的系数数量是16个，去掉DC项就是15个
    features_rest = torch.zeros(
        (num_gaussians, num_rest_coeffs, 3),
        dtype=features_dc.dtype,
        device=features_dc.device,
    )
    return GaussianParameters(
        means=means,
        features_dc=features_dc,
        features_rest = features_rest,
        opacity_logits=opacity_logits,
        rotations=rotations,
        log_scales=log_scales,
    )







def initial_opacity_logits(initialOpacityValue : float, num_gaussians : int) -> torch.Tensor:
    r"""
    初始化每个 Gaussian 的不透明度 logit 参数。

    真实不透明度 alpha 必须在 (0, 1) 内。训练时不直接优化 alpha，
    而是优化它的 logit 值；渲染时再通过 sigmoid 转回 alpha。

    公式：

    $$
    o = \log \frac{\alpha}{1 - \alpha}
    $$

    Args:
        initial_opacity_value: 初始真实不透明度 alpha，例如 0.1。
        num_gaussians: Gaussian 数量。

    Returns:
        opacity_logits: shape 为 [num_gaussians, 1] 的不透明度 logit 张量。
    """
    opacity_logits = _inverse_sigmoid(
        torch.full((num_gaussians, 1), initialOpacityValue)
    )
    return opacity_logits



def initial_rotations(num_gaussians: int) -> torch.Tensor:
    """
    初始化每个 Gaussian 的旋转四元数。

    3DGS 使用四元数 q = (w, x, y, z) 表示旋转。
    单位四元数 (1, 0, 0, 0) 表示不旋转，因此所有 Gaussian 初始都设为单位旋转。

    Args:
        num_gaussians: Gaussian 数量。

    Returns:
        rotations: shape 为 [num_gaussians, 4] 的旋转四元数张量。
    """
    rotations = torch.zeros((num_gaussians, 4), dtype=torch.float32)
    rotations[:, 0] = 1.0
    return rotations


def initial_log_scales(initial_scale_value: float, num_gaussians: int) -> torch.Tensor:
    """
    初始化每个 Gaussian 的 log scale 参数。

    真实尺度必须为正数。训练时保存的是 log(scale)，渲染时通过 exp(log_scale)
    转回真实尺度，从而保证 scale 始终大于 0。

    公式:
        log_scale = log(scale)

    Args:
        initial_scale_value: 初始真实尺度，例如 0.01。
        num_gaussians: Gaussian 数量。

    Returns:
        log_scales: shape 为 [num_gaussians, 3] 的 log scale 张量。
    """
    scales = torch.full((num_gaussians, 3), initial_scale_value)
    log_scales = torch.log(scales)
    return log_scales


def initial_log_scales_from_nearest_neighbors(
    means: torch.Tensor,
    num_neighbors: int = 3,
    min_dist2: float = 1e-7,
    chunk_size: int = 1024,
) -> torch.Tensor:
    r"""
    用点云最近邻距离初始化 Gaussian 的 log scale。

    原版 3DGS 不是把所有 Gaussian 都初始化成固定大小，而是让每个 Gaussian
    的初始大小和局部点云稀疏程度相关。点越稀疏，初始 Gaussian 越大；
    点越密集，初始 Gaussian 越小。

    对第 \(i\) 个点，计算它到最近 \(k\) 个点的平方距离均值：

    $$
    d_i^2
    =
    \frac{1}{k}
    \sum_{j \in \mathcal{N}_k(i)}
    \lVert x_i - x_j \rVert^2
    $$

    然后初始化：

    $$
    s_i = \sqrt{\max(d_i^2, \epsilon)}
    $$

    $$
    \ell_i = \log(s_i)
    $$

    Args:
        means: shape 为 [N, 3] 的点云坐标。
        num_neighbors: 最近邻数量。原版 simple-knn 用 3。
        min_dist2: 最小平方距离，避免 log(0)。
        chunk_size: 分块计算距离，避免一次性构造 [N, N] 距离矩阵。

    Returns:
        log_scales: shape 为 [N, 3] 的 log scale。
    """
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape [N, 3]")

    num_points = means.shape[0]
    if num_points <= num_neighbors:
        return initial_log_scales(0.01, num_points)

    means_for_distance = means.detach().float()
    nearest_dist2_chunks = []

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        chunk = means_for_distance[start:end]

        dist2 = torch.cdist(chunk, means_for_distance, p=2) ** 2
        row_ids = torch.arange(end - start, device=dist2.device)
        dist2[row_ids, torch.arange(start, end, device=dist2.device)] = float("inf")

        nearest_dist2 = torch.topk(
            dist2,
            k=num_neighbors,
            dim=1,
            largest=False,
        ).values
        nearest_dist2_chunks.append(nearest_dist2.mean(dim=1))

    mean_dist2 = torch.cat(nearest_dist2_chunks, dim=0).clamp_min(min_dist2)
    scales = torch.sqrt(mean_dist2)[:, None].repeat(1, 3)
    return torch.log(scales).to(dtype=means.dtype, device=means.device)

def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))



if __name__ == "__main__":
    data = main_initialize_gaussians_from_point_cloud("src/datasets/SeathruNeRF_dataset/Curasao/sparse/0/points3D.bin")
    print(data.features_dc.shape)
    print(data.features_rest.shape)
