import torch
from torch import nn
from methods.leo_3DGS.contracts.gaussian import GaussianParameters

class GaussianModel(nn.Module):
    def __init__(self, params: GaussianParameters):
        super().__init__()

        self.means = nn.Parameter(params.means)
        self.colors = nn.Parameter(params.colors)
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