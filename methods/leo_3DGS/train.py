from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from methods.leo_3DGS.functions.initialGS import main_initialize_gaussians_from_point_cloud
from methods.leo_3DGS.functions.render import GaussianModel, render_original_cuda
from methods.leo_3DGS.utils_.loadData import (
    load_colmap_cameras_bin, 
    load_colmap_images_bin, 
    load_gt_image,
    compute_scene_extent_from_colmap_images
)
from methods.leo_3DGS.adapters.densification.raw3DGS import (
    accumulate_densification_stats,
    create_densification_state,
    densification_step,
    reset_opacity,
)

def create_optimizer(model: GaussianModel):
    return torch.optim.Adam(
        [
            {"params": [model.means], "lr": 1.6e-4},
            {"params": [model.features_dc], "lr": 2.5e-3},
            {"params": [model.features_rest], "lr": 1.25e-4},
            {"params": [model.opacity_logits], "lr": 5.0e-2},
            {"params": [model.log_scales], "lr": 5.0e-3},
            {"params": [model.rotations], "lr": 1.0e-3},
        ],
        eps=1e-15,
    )

def save_render(rendered: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    image = rendered.detach().clamp(0.0, 1.0)
    image = image.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    image = (image.cpu().numpy() * 255.0).astype(np.uint8)

    Image.fromarray(image).save(path)

def main():
    device = torch.device("cuda")
    dataset_root = Path("src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole")
    sparse_root = dataset_root / "sparse" / "0"
    image_root = dataset_root / "images"
    output_root = Path("outputs/leo_3DGS/train_one_image")
    
    data = main_initialize_gaussians_from_point_cloud(str(sparse_root / "points3D.bin"))
    model = GaussianModel(data).to(device)
    model.train()
    
    cameras_para = load_colmap_cameras_bin(sparse_root / "cameras.bin")
    images_para = load_colmap_images_bin(sparse_root / "images.bin")
    
    image_para = sorted(images_para.values(), key=lambda x: x.name)[0]
    camera_para = cameras_para[image_para.camera_id]
    
    label_image = load_gt_image(image_root / image_para.name, camera_para, device)
    
    optimizer = create_optimizer(model)
    bg_color = torch.zeros(3, device=device)
    densify_state = create_densification_state(model, percent_dense=0.01)
    scene_extent = compute_scene_extent_from_colmap_images(images_para)
    
    for iteration in range(1, 20001):
        pkg = render_original_cuda(
            model=model,
            image=image_para,
            camera=camera_para,
            bg_color=bg_color,
            sh_degree=3,
        )
        
        rendered = pkg["render"]  # [3, H, W]
        
        loss = F.l1_loss(rendered, label_image)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        
        accumulate_densification_stats(densify_state, pkg)

        if iteration > 500 and iteration % 100 == 0 and iteration < 9000:
            densify_state = densification_step(
                model=model,
                optimizer=optimizer,
                state=densify_state,
                scene_extent=scene_extent,
                max_grad=0.0002,
                min_opacity=0.005,
                max_screen_size=None,
            )

        if iteration % 3000 == 0 and iteration < 6001:
            reset_opacity(model, optimizer)
        
        optimizer.step()
        
        
        
        
        # with torch.no_grad():
        #     model.opacity_logits.clamp_(min=-10.0, max=10.0)
        #     model.log_scales.clamp_(min=-8.0, max=2.0)

        if iteration == 1 or iteration % 1000 == 0:
            out_path = output_root / f"render_{iteration:04d}.png"
            save_render(rendered, out_path)
            print(f"iter={iteration:04d}, loss={loss.item():.6f}, saved={out_path}")




if __name__ == "__main__":
    
    
    main()
   
