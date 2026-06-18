from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GaussianParameters:
    means: torch.Tensor
    colors: torch.Tensor
    opacity_logits: torch.Tensor
    rotations: torch.Tensor
    log_scales: torch.Tensor