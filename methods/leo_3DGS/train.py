from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from methods.leo_3DGS.functions.metrics import LPIPSEvaluator, evaluate_image_metrics
from methods.leo_3DGS.functions.initialGS import main_initialize_gaussians_from_point_cloud
from methods.leo_3DGS.functions.render import GaussianModel, render_original_cuda
from methods.leo_3DGS.functions.metrics import save_gaussian_projection_debug
from methods.leo_3DGS.adapters.loss.raw3DGS import raw_3dgs_loss
from methods.leo_3DGS.adapters.loss.adjustraw3DGS import adjust_raw_3dgs_loss
from methods.leo_3DGS.utils_.loadData import (
    ColmapCameraData,
    ColmapImageData,
    compute_scene_extent_from_colmap_images,
    load_colmap_cameras_bin,
    load_colmap_images_bin,
    load_gt_image,
)
from methods.leo_3DGS.adapters.densification.raw3DGS import (
    accumulate_densification_stats,
    create_densification_state,
    densification_step,
    reset_opacity,
)
from methods.leo_3DGS.functions.metrics import LPIPSEvaluator
from methods.leo_3DGS.adapters.loss.adjustraw3DGS import (
    LPIPSLoss,
    adjust_raw_3dgs_loss,
)

@dataclass(frozen=True)
class TrainingView:
    image: ColmapImageData
    camera: ColmapCameraData
    gt_image: torch.Tensor


def create_optimizer(model: GaussianModel):
    position_lr = 1.6e-4
    feature_lr = 2.5e-3
    opacity_lr = 5.0e-2
    scaling_lr = 5.0e-3
    rotation_lr = 1.0e-3

    return torch.optim.Adam(
        [
            {"params": [model.means], "lr": position_lr},
            {"params": [model.features_dc], "lr": feature_lr},
            {"params": [model.features_rest], "lr": feature_lr / 20.0},
            {"params": [model.opacity_logits], "lr": opacity_lr},
            {"params": [model.log_scales], "lr": scaling_lr},
            {"params": [model.rotations], "lr": rotation_lr},
        ],
        eps=1e-15,
    )


def save_render(rendered: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    image = rendered.detach().clamp(0.0, 1.0)
    image = image.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    image = (image.cpu().numpy() * 255.0).astype(np.uint8)

    Image.fromarray(image).save(path)


def scaled_camera(camera: ColmapCameraData, max_image_size: int | None) -> ColmapCameraData:
    if max_image_size is None or max_image_size <= 0:
        return camera

    longest_side = max(camera.width, camera.height)
    scale = min(1.0, float(max_image_size) / float(longest_side))

    if scale == 1.0:
        return camera

    if camera.model == "PINHOLE":
        params = camera.params.copy()
        params[:4] *= scale
    elif camera.model == "SIMPLE_PINHOLE":
        params = camera.params.copy()
        params[:3] *= scale
    else:
        raise ValueError(f"image scaling only supports PINHOLE/SIMPLE_PINHOLE cameras, got {camera.model}")

    return ColmapCameraData(
        camera_id=camera.camera_id,
        model=camera.model,
        width=int(round(camera.width * scale)),
        height=int(round(camera.height * scale)),
        params=params,
    )


def build_view_cache(
    sparse_root: Path,
    image_root: Path,
    device: torch.device,
    max_image_size: int | None = 1600,
    test_every: int = 8,
    cache_images_on_gpu: bool = True,
) -> tuple[list[TrainingView], list[TrainingView], list[ColmapImageData]]:
    cameras_para = load_colmap_cameras_bin(sparse_root / "cameras.bin")
    images_para = load_colmap_images_bin(sparse_root / "images.bin")
    all_images = sorted(images_para.values(), key=lambda x: x.name)

    if not all_images:
        raise ValueError("COLMAP images.bin contains no images")

    image_device = device if cache_images_on_gpu else torch.device("cpu")
    views: list[TrainingView] = []

    for image_para in all_images:
        camera_para = scaled_camera(cameras_para[image_para.camera_id], max_image_size)
        gt_image = load_gt_image(image_root / image_para.name, camera_para, image_device)

        if not cache_images_on_gpu and device.type == "cuda":
            gt_image = gt_image.pin_memory()

        views.append(
            TrainingView(
                image=image_para,
                camera=camera_para,
                gt_image=gt_image,
            )
        )

    if test_every <= 0:
        train_views = views
        test_views = []
    else:
        test_views = views[::test_every]
        train_views = [view for i, view in enumerate(views) if i % test_every != 0]

    if not train_views:
        raise ValueError("train_views is empty; use a larger dataset or set test_every <= 0")

    return train_views, test_views, all_images


def move_gt_to_device(view: TrainingView, device: torch.device) -> torch.Tensor:
    gt_device = view.gt_image.device
    if gt_device.type == device.type and (device.index is None or gt_device.index == device.index):
        return view.gt_image
    return view.gt_image.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_test_set(
    model: GaussianModel,
    test_views: list[TrainingView],
    device: torch.device,
    bg_color: torch.Tensor,
    lpips_evaluator: LPIPSEvaluator,
    output_root: Path,
    global_step: int,
    save_first_render: bool = True,
    save_first_projection_debug: bool = True,
) -> dict[str, float]:
    if not test_views:
        return {}

    was_training = model.training
    model.eval()

    l1_values = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for index, view in enumerate(test_views):
        label_image = move_gt_to_device(view, device)
        pkg = render_original_cuda(
            model=model,
            image=view.image,
            camera=view.camera,
            bg_color=bg_color,
            sh_degree=3,
        )
        rendered = pkg["render"]

        if rendered.shape != label_image.shape:
            raise RuntimeError(
                f"test render shape {tuple(rendered.shape)} does not match label shape {tuple(label_image.shape)}"
            )

        l1_values.append(F.l1_loss(rendered, label_image).item())
        metrics = evaluate_image_metrics(
            rendered=rendered,
            target=label_image,
            lpips_evaluator=lpips_evaluator,
        )
        psnr_values.append(metrics.psnr)
        ssim_values.append(metrics.ssim)
        lpips_values.append(metrics.lpips)

        if save_first_render and index == 1:
            save_render(
                rendered,
                output_root / f"test_render_step_{global_step:06d}_{view.image.name}",
            )

        if save_first_projection_debug and index == 1:
            save_gaussian_projection_debug(
                model=model,
                image=view.image,
                camera=view.camera,
                background=label_image,
                output_path=output_root / f"test_projection_step_{global_step:06d}_{view.image.name}",
                point_color=(255, 0, 0),
                point_radius=1,
                stride=1,
            )

    if was_training:
        model.train()

    return {
        "l1": float(np.mean(l1_values)),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "lpips": float(np.mean(lpips_values)),
    }


def main():
    device = torch.device("cuda")
    dataset_root = Path("src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole")
    sparse_root = dataset_root / "sparse" / "0"
    image_root = dataset_root / "images"
    output_root = Path("outputs/leo_3DGS/train_views")

    max_epochs = 20000
    max_image_size = 1600
    test_every = 8
    cache_images_on_gpu = True
    save_interval = 1000
    test_interval = 1000
    densify_from_iter = 500
    densify_until_iter = 9000
    densification_interval = 100
    opacity_reset_interval = 3000
    opacity_reset_until = 6000

    data = main_initialize_gaussians_from_point_cloud(str(sparse_root / "points3D.bin"))
    model = GaussianModel(data).to(device)
    model.train()

    train_views, test_views, all_images = build_view_cache(
        sparse_root=sparse_root,
        image_root=image_root,
        device=device,
        max_image_size=max_image_size,
        test_every=test_every,
        cache_images_on_gpu=cache_images_on_gpu,
    )
    
    # #下面这两行会只训练一张图片
    # train_views = [train_views[0]]
    # test_views = []

    lpips_evaluator = LPIPSEvaluator(net_type="vgg", device=device)
    lpips_loss_model = LPIPSLoss(net_type="vgg").to(device).eval()   # 训练 loss 用的 LPIPS 模型，和评测用的可以是同一个，也可以不同（比如评测用更大更慢的模型）
    
    optimizer = create_optimizer(model)
    # bg_color = torch.zeros(3, device=device)
    bg_color = torch.tensor([0.1443, 0.1867, 0.2528], device=device)
    densify_state = create_densification_state(model, percent_dense=0.01)
    scene_extent = compute_scene_extent_from_colmap_images([view.image for view in train_views])

    print(
        f"train_views={len(train_views)}, test_views={len(test_views)}, "
        f"total_images={len(all_images)}, max_image_size={max_image_size}"
    )

    global_step = 0

    for epoch in range(1, max_epochs + 1):
        epoch_views = train_views.copy()
        random.shuffle(epoch_views)

        for view in epoch_views:
            global_step += 1
            label_image = move_gt_to_device(view, device)

            pkg = render_original_cuda(
                model=model,
                image=view.image,
                camera=view.camera,
                bg_color=bg_color,
                sh_degree=3,
            )

            rendered = pkg["render"]  # [3, H, W]
            if rendered.shape != label_image.shape:
                raise RuntimeError(
                    f"render shape {tuple(rendered.shape)} does not match label shape {tuple(label_image.shape)}"
                )

            # loss_result = raw_3dgs_loss(
            #     rendered=rendered,
            #     target=label_image,
            #     lambda_dssim=0.2,
            # )
            loss_result = adjust_raw_3dgs_loss(
                rendered=rendered,
                target=label_image,
                lambda_dssim=0.2,
                lambda_lpips=0.02,
                lpips_loss_model=lpips_loss_model,
            )
            loss = loss_result.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if densify_from_iter < global_step < densify_until_iter:
                accumulate_densification_stats(densify_state, pkg)

            if (
                densify_from_iter < global_step < densify_until_iter
                and global_step % densification_interval == 0
            ):
                densify_state = densification_step(
                    model=model,
                    optimizer=optimizer,
                    state=densify_state,
                    scene_extent=scene_extent,
                    max_grad=0.0002,
                    min_opacity=0.005,
                    max_screen_size=30,
                )

            if global_step % opacity_reset_interval == 0 and global_step <= opacity_reset_until:
                reset_opacity(model, optimizer)

            optimizer.step()
            
            if global_step == 1 or global_step % 100 == 0:
                metrics = evaluate_image_metrics(
                    rendered=rendered,
                    target=label_image,
                    lpips_evaluator=lpips_evaluator,
                )
                print(
                    "step", global_step,
                    "loss", loss.item(),
                    "num_gaussians", model.means.shape[0],
                    "visible", pkg["visibility_filter"].sum().item(),
                    "alpha_mean", pkg["alpha"].mean().item(),
                    "alpha_max", pkg["alpha"].max().item(),
                    f"psnr={metrics.psnr:.4f}, "
                    f"ssim={metrics.ssim:.4f}, "
                    f"lpips={metrics.lpips:.4f}, "
                )
                

            if global_step == 1 or global_step % save_interval == 0:
                out_path = output_root / f"render_step_{global_step:06d}.png"
                save_render(rendered, out_path)
                print(
                    f"epoch={epoch:04d}, step={global_step:06d}, "
                    f"image={view.image.name}, loss={loss.item():.6f}, saved={out_path}"
                )
                save_gaussian_projection_debug(
                    model=model,
                    image=view.image,
                    camera=view.camera,
                    background=label_image,
                    output_path=output_root / f"projection_step_{global_step:06d}.png",
                    point_color=(255, 0, 0),
                    point_radius=1,
                    stride=1,
                )

            if global_step == 1 or global_step % test_interval == 0:
                test_metrics = evaluate_test_set(
                    model=model,
                    test_views=test_views,
                    device=device,
                    bg_color=bg_color,
                    lpips_evaluator=lpips_evaluator,
                    output_root=output_root,
                    global_step=global_step,
                    save_first_render=True,
                    save_first_projection_debug=True,
                )
                if test_metrics:
                    print(
                        f"test step={global_step:06d}, "
                        f"l1={test_metrics['l1']:.6f}, "
                        f"psnr={test_metrics['psnr']:.4f}, "
                        f"ssim={test_metrics['ssim']:.4f}, "
                        f"lpips={test_metrics['lpips']:.4f}"
                    )
                            
if __name__ == "__main__":
    main()
