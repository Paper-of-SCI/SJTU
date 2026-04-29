"""utils 包：框架无关的数据 I/O 与数学工具。

所有模块仅依赖 NumPy、Pillow、scipy 和标准库，无 PyTorch/JAX 依赖。
"""

from utils.camera_utils import (
    qvec_to_rotmat,
    rotmat_to_qvec,
    build_intrinsic_matrix,
    build_projection_matrix,
    focal_to_fov,
    fov_to_focal,
    colmap_to_opengl_c2w,
    w2c_to_c2w,
    c2w_to_w2c,
    interpolate_camera_path,
)

from utils.colmap_reader import (
    COLMAPCamera,
    COLMAPImage,
    COLMAPPoint3D,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
    read_points3D_text,
    read_colmap_model,
    write_colmap_model,
)

from utils.ply_io import (
    read_ply,
    write_ply,
    gaussians_to_ply_dict,
    ply_dict_to_gaussians,
)

from utils.image_utils import (
    load_image,
    save_image,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    images_to_video,
    linear_to_srgb,
    srgb_to_linear,
)

from utils.dataset_loaders import (
    SceneData,
    load_colmap_dataset,
    load_llff_dataset,
    load_blender_dataset,
    compute_scene_scale,
)

__all__ = [
    # camera_utils
    "qvec_to_rotmat", "rotmat_to_qvec",
    "build_intrinsic_matrix", "build_projection_matrix",
    "focal_to_fov", "fov_to_focal",
    "colmap_to_opengl_c2w", "w2c_to_c2w", "c2w_to_w2c",
    "interpolate_camera_path",
    # colmap_reader
    "COLMAPCamera", "COLMAPImage", "COLMAPPoint3D",
    "read_cameras_binary", "read_cameras_text",
    "read_images_binary", "read_images_text",
    "read_points3D_binary", "read_points3D_text",
    "read_colmap_model", "write_colmap_model",
    # ply_io
    "read_ply", "write_ply",
    "gaussians_to_ply_dict", "ply_dict_to_gaussians",
    # image_utils
    "load_image", "save_image",
    "compute_psnr", "compute_ssim", "compute_lpips",
    "images_to_video", "linear_to_srgb", "srgb_to_linear",
    # dataset_loaders
    "SceneData",
    "load_colmap_dataset", "load_llff_dataset", "load_blender_dataset",
    "compute_scene_scale",
]
