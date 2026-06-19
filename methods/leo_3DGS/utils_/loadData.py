from __future__ import annotations

from methods.leo_3DGS.adapters.colmap_data import (
    ColmapCameraData,
    ColmapImageData,
    PointCloudData,
    colmap_image_camera_center,
    compute_scene_extent_from_colmap_images,
    load_colmap_cameras_bin,
    load_colmap_images_bin,
    load_colmap_points3d_bin,
    load_gt_image,
)

__all__ = [
    "ColmapCameraData",
    "ColmapImageData",
    "PointCloudData",
    "colmap_image_camera_center",
    "compute_scene_extent_from_colmap_images",
    "load_colmap_cameras_bin",
    "load_colmap_images_bin",
    "load_colmap_points3d_bin",
    "load_gt_image",
]


if __name__ == "__main__":
    cameras = load_colmap_cameras_bin("src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole/sparse/0/cameras.bin")
    for camera in cameras.values():
        print(camera.camera_id, camera.model, camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy)

    images = load_colmap_images_bin("src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole/sparse/0/images.bin")
    print("images:", len(images))
    for image in sorted(images.values(), key=lambda item: item.name)[:]:
        print(image.image_id, image.name, image.camera_id, image.qvec, image.tvec)
