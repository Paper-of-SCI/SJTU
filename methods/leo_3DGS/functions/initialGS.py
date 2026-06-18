from methods.leo_3DGS.utils_.loadData import load_colmap_points3d_bin
import torch
from methods.leo_3DGS.contracts.gaussian import GaussianParameters

'''
Gaussians球有以下参数

means          -> 高斯中心                   [N, 3]
colors         -> 初始颜色                   [N, 3]
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
    初始化高斯球的scale为一个较小的值，例如0.01。因为scale必须是正数，所以我们训练log_scale，初始值是log(0.01)。
    '''
    log_scales = initial_log_scales(0.01, num_gaussians)


    return GaussianParameters(
        means=means,
        colors=colors,
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

def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))



if __name__ == "__main__":
    data = main_initialize_gaussians_from_point_cloud("src/datasets/SeathruNeRF_dataset/Curasao/sparse/0/points3D.bin")
    print( data.rotations[233], data.log_scales[232])