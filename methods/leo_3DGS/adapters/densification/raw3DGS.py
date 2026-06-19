from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from methods.leo_3DGS.functions.render import GaussianModel


PARAMETER_SPECS = {
    "xyz": "means",
    "f_dc": "features_dc",
    "f_rest": "features_rest",
    "opacity": "opacity_logits",
    "scaling": "log_scales",
    "rotation": "rotations",
}

PARAMETER_ALIASES = {
    "xyz": "xyz",
    "means": "xyz",
    "f_dc": "f_dc",
    "features_dc": "f_dc",
    "f_rest": "f_rest",
    "features_rest": "f_rest",
    "opacity": "opacity",
    "opacity_logits": "opacity",
    "scales": "scaling",
    "scaling": "scaling",
    "log_scales": "scaling",
    "rotation": "rotation",
    "rotations": "rotation",
}


@dataclass
class DensificationState:
    xyz_gradient_accum: torch.Tensor
    denom: torch.Tensor
    max_radii2D: torch.Tensor
    percent_dense: float = 0.01


def create_densification_state(
    model: GaussianModel,
    percent_dense: float = 0.01,
) -> DensificationState:
    num_gaussians = model.means.shape[0]
    device = model.means.device
    dtype = model.means.dtype

    return DensificationState(
        xyz_gradient_accum=torch.zeros((num_gaussians, 1), dtype=dtype, device=device),
        denom=torch.zeros((num_gaussians, 1), dtype=dtype, device=device),
        max_radii2D=torch.zeros((num_gaussians,), dtype=torch.int32, device=device),
        percent_dense=float(percent_dense),
    )


def add_densification_stats(
    state: DensificationState,
    viewspace_points: torch.Tensor,
    visibility_filter: torch.Tensor,
) -> None:
    if viewspace_points.grad is None:
        raise RuntimeError("viewspace_points.grad is None; call loss.backward() before add_densification_stats")

    state.xyz_gradient_accum[visibility_filter] += torch.norm(
        viewspace_points.grad[visibility_filter, :2],
        dim=-1,
        keepdim=True,
    )
    state.denom[visibility_filter] += 1


def update_max_radii2D(
    state: DensificationState,
    visibility_filter: torch.Tensor,
    radii: torch.Tensor,
) -> None:
    state.max_radii2D[visibility_filter] = torch.max(
        state.max_radii2D[visibility_filter],
        radii[visibility_filter].to(state.max_radii2D.dtype),
    )


def accumulate_densification_stats(
    state: DensificationState,
    render_pkg: dict[str, torch.Tensor],
) -> None:
    """
    每轮 backward 后累计原版 3DGS densification 需要的统计量。

    这一步应每轮执行；真正 clone/split/prune 可以按 interval 条件执行。
    """
    with torch.no_grad():
        visibility_filter = render_pkg["visibility_filter"]

        update_max_radii2D(
            state=state,
            visibility_filter=visibility_filter,
            radii=render_pkg["radii"],
        )

        add_densification_stats(
            state=state,
            viewspace_points=render_pkg["viewspace_points"],
            visibility_filter=visibility_filter,
        )


def densification_step(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    state: DensificationState,
    scene_extent: float,
    max_grad: float = 0.0002,
    min_opacity: float = 0.005,
    max_screen_size: int | None = None,
    split_samples: int = 2,
) -> DensificationState:
    """
    执行一次原版 3DGS 风格的 clone / split / prune。

    调用时机：`loss.backward()` 之后，`optimizer.step()` 之前。

    注意：调用这个函数之前，应先每轮调用 `accumulate_densification_stats()`。

    Args:
        model: GaussianModel。
        optimizer: 管理 Gaussian 参数的 optimizer。
        state: densification 状态。
        scene_extent: 原版里的 `scene.cameras_extent`。
        max_grad: viewspace 梯度阈值。
        min_opacity: prune 的 opacity 下限。
        max_screen_size: 屏幕半径过大的 prune 阈值。
        split_samples: split 一个 Gaussian 时生成几个新 Gaussian。

    Returns:
        更新后的 DensificationState。
    """
    with torch.no_grad():
        return densify_and_prune(
            model=model,
            optimizer=optimizer,
            state=state,
            max_grad=max_grad,
            min_opacity=min_opacity,
            extent=scene_extent,
            max_screen_size=max_screen_size,
            split_samples=split_samples,
        )


def densify_and_prune(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    state: DensificationState,
    max_grad: float,
    min_opacity: float,
    extent: float,
    max_screen_size: int | None = None,
    split_samples: int = 2,
) -> DensificationState:
    grads = state.xyz_gradient_accum / state.denom
    grads[grads.isnan()] = 0.0

    state = densify_and_clone(
        model=model,
        optimizer=optimizer,
        state=state,
        grads=grads,
        grad_threshold=max_grad,
        scene_extent=extent,
    )

    state = densify_and_split(
        model=model,
        optimizer=optimizer,
        state=state,
        grads=grads,
        grad_threshold=max_grad,
        scene_extent=extent,
        split_samples=split_samples,
    )

    prune_mask = (model.opacities < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = state.max_radii2D > max_screen_size
        big_points_ws = model.scales.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

    state = prune_points(
        model=model,
        optimizer=optimizer,
        state=state,
        prune_mask=prune_mask,
    )

    torch.cuda.empty_cache()
    return state


def densify_and_clone(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    state: DensificationState,
    grads: torch.Tensor,
    grad_threshold: float,
    scene_extent: float,
) -> DensificationState:
    selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
    selected_pts_mask = torch.logical_and(
        selected_pts_mask,
        model.scales.max(dim=1).values <= state.percent_dense * scene_extent,
    )

    new_params = {
        "xyz": model.means[selected_pts_mask],
        "f_dc": model.features_dc[selected_pts_mask],
        "f_rest": model.features_rest[selected_pts_mask],
        "opacity": model.opacity_logits[selected_pts_mask],
        "scaling": model.log_scales[selected_pts_mask],
        "rotation": model.rotations[selected_pts_mask],
    }

    return densification_postfix(model, optimizer, state, new_params)


def densify_and_split(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    state: DensificationState,
    grads: torch.Tensor,
    grad_threshold: float,
    scene_extent: float,
    split_samples: int = 2,
) -> DensificationState:
    n_init_points = model.means.shape[0]
    padded_grad = torch.zeros((n_init_points,), dtype=model.means.dtype, device=model.means.device)
    padded_grad[: grads.shape[0]] = grads.squeeze()

    selected_pts_mask = padded_grad >= grad_threshold
    selected_pts_mask = torch.logical_and(
        selected_pts_mask,
        model.scales.max(dim=1).values > state.percent_dense * scene_extent,
    )

    stds = model.scales[selected_pts_mask].repeat(split_samples, 1)
    sample_means = torch.zeros((stds.shape[0], 3), dtype=model.means.dtype, device=model.means.device)
    samples = torch.normal(mean=sample_means, std=stds)

    rotations = quaternion_to_rotmat_batch(model.rotations[selected_pts_mask]).repeat(split_samples, 1, 1)
    new_xyz = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
    new_xyz = new_xyz + model.means[selected_pts_mask].repeat(split_samples, 1)

    new_scaling = torch.log(
        model.scales[selected_pts_mask].repeat(split_samples, 1) / (0.8 * split_samples)
    )
    new_rotation = model.rotations[selected_pts_mask].repeat(split_samples, 1)
    new_features_dc = model.features_dc[selected_pts_mask].repeat(split_samples, 1, 1)
    new_features_rest = model.features_rest[selected_pts_mask].repeat(split_samples, 1, 1)
    new_opacity = model.opacity_logits[selected_pts_mask].repeat(split_samples, 1)

    state = densification_postfix(
        model=model,
        optimizer=optimizer,
        state=state,
        new_params={
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        },
    )

    prune_filter = torch.cat(
        (
            selected_pts_mask,
            torch.zeros(
                split_samples * selected_pts_mask.sum(),
                dtype=torch.bool,
                device=model.means.device,
            ),
        )
    )
    return prune_points(model, optimizer, state, prune_filter)


def prune_points(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    state: DensificationState,
    prune_mask: torch.Tensor,
) -> DensificationState:
    valid_points_mask = ~prune_mask
    pruned_tensors = prune_optimizer(model, optimizer, valid_points_mask)
    set_model_parameters(model, pruned_tensors)

    return DensificationState(
        xyz_gradient_accum=state.xyz_gradient_accum[valid_points_mask],
        denom=state.denom[valid_points_mask],
        max_radii2D=state.max_radii2D[valid_points_mask],
        percent_dense=state.percent_dense,
    )


def reset_opacity(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    max_opacity: float = 0.01,
) -> None:
    target_opacity = torch.min(
        model.opacities,
        torch.ones_like(model.opacities) * max_opacity,
    )
    new_opacity_logits = inverse_sigmoid(target_opacity)
    replaced_tensors = replace_tensor_to_optimizer(
        model=model,
        optimizer=optimizer,
        tensor=new_opacity_logits,
        name="opacity",
    )
    set_model_parameters(model, replaced_tensors)


def densification_postfix(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    state: DensificationState,
    new_params: dict[str, torch.Tensor],
) -> DensificationState:
    optimizable_tensors = cat_tensors_to_optimizer(model, optimizer, new_params)
    set_model_parameters(model, optimizable_tensors)

    num_gaussians = model.means.shape[0]
    return DensificationState(
        xyz_gradient_accum=torch.zeros((num_gaussians, 1), dtype=model.means.dtype, device=model.means.device),
        denom=torch.zeros((num_gaussians, 1), dtype=model.means.dtype, device=model.means.device),
        max_radii2D=torch.zeros((num_gaussians,), dtype=torch.int32, device=model.means.device),
        percent_dense=state.percent_dense,
    )


def cat_tensors_to_optimizer(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    tensors_dict: dict[str, torch.Tensor],
) -> dict[str, nn.Parameter]:
    optimizable_tensors: dict[str, nn.Parameter] = {}

    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        name = get_param_group_name(model, group)
        extension_tensor = tensors_dict[name]
        old_parameter = group["params"][0]
        stored_state = optimizer.state.get(old_parameter, None)
        requires_grad = old_parameter.requires_grad

        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                dim=0,
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                dim=0,
            )

            del optimizer.state[old_parameter]
            group["params"][0] = nn.Parameter(
                torch.cat((old_parameter, extension_tensor), dim=0).requires_grad_(requires_grad),
                requires_grad=requires_grad,
            )
            optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                torch.cat((old_parameter, extension_tensor), dim=0).requires_grad_(requires_grad),
                requires_grad=requires_grad,
            )

        optimizable_tensors[name] = group["params"][0]

    return optimizable_tensors


def prune_optimizer(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    valid_points_mask: torch.Tensor,
) -> dict[str, nn.Parameter]:
    optimizable_tensors: dict[str, nn.Parameter] = {}

    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        name = get_param_group_name(model, group)
        old_parameter = group["params"][0]
        stored_state = optimizer.state.get(old_parameter, None)
        requires_grad = old_parameter.requires_grad

        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][valid_points_mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_points_mask]

            del optimizer.state[old_parameter]
            group["params"][0] = nn.Parameter(
                old_parameter[valid_points_mask].requires_grad_(requires_grad),
                requires_grad=requires_grad,
            )
            optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                old_parameter[valid_points_mask].requires_grad_(requires_grad),
                requires_grad=requires_grad,
            )

        optimizable_tensors[name] = group["params"][0]

    return optimizable_tensors


def replace_tensor_to_optimizer(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    tensor: torch.Tensor,
    name: str,
) -> dict[str, nn.Parameter]:
    canonical_name = normalize_parameter_name(name)
    optimizable_tensors: dict[str, nn.Parameter] = {}

    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        group_name = get_param_group_name(model, group)
        if group_name != canonical_name:
            continue

        old_parameter = group["params"][0]
        stored_state = optimizer.state.get(old_parameter, None)
        requires_grad = old_parameter.requires_grad

        if stored_state is not None:
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del optimizer.state[old_parameter]
            group["params"][0] = nn.Parameter(
                tensor.requires_grad_(requires_grad),
                requires_grad=requires_grad,
            )
            optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                tensor.requires_grad_(requires_grad),
                requires_grad=requires_grad,
            )

        optimizable_tensors[group_name] = group["params"][0]

    return optimizable_tensors


def set_model_parameters(model: GaussianModel, tensors: dict[str, nn.Parameter]) -> None:
    for name, parameter in tensors.items():
        attr_name = PARAMETER_SPECS[name]
        setattr(model, attr_name, parameter)


def get_param_group_name(model: GaussianModel, group: dict) -> str:
    if "name" in group:
        return normalize_parameter_name(group["name"])

    parameter = group["params"][0]
    for name, attr_name in PARAMETER_SPECS.items():
        if getattr(model, attr_name) is parameter:
            group["name"] = name
            return name

    raise ValueError("optimizer param group does not match GaussianModel parameters")


def normalize_parameter_name(name: str) -> str:
    if name not in PARAMETER_ALIASES:
        raise ValueError(f"unknown optimizer parameter group name: {name}")
    return PARAMETER_ALIASES[name]


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def quaternion_to_rotmat_batch(q: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(q, dim=1)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    row0 = torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        dim=1,
    )

    row1 = torch.stack(
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        dim=1,
    )

    row2 = torch.stack(
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dim=1,
    )

    return torch.stack([row0, row1, row2], dim=1)
