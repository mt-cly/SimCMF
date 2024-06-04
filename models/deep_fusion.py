import torch
import torch.nn as nn
from functools import partial
import torch.utils.checkpoint as cp
# from ops.modules import MSDeformAttn

def get_deep_fusion_blocks(num_layers, SAM_embedding_chans, proj_type, pretrain=None):
    blocks = nn.ModuleList()
    for i in range(num_layers):
        blocks.append(_create_deep_block(i, SAM_embedding_chans, proj_type, pretrain))
    return blocks

def _create_deep_block(layer_id, SAM_embedding_chans, proj_type, pretrain=None):
    return Identity_Block()


class Identity_Block(nn.Module):
    def __init__(self):
        super(Identity_Block, self).__init__()

    def forward(self, rgb_embedding, x_embedding):
        return rgb_embedding, x_embedding


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

