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
    if proj_type == 'vipt_shallow':
        if layer_id == 0:
            return ViPT_Block(inplanes=SAM_embedding_chans, hide_channel=8, smooth=True)
        else:
            return Identity_Block()
    elif proj_type == 'vipt_deep':
        return ViPT_Block(inplanes=SAM_embedding_chans, hide_channel=8, smooth=True)
    elif proj_type == 'cmx':
        return CMX_Block(SAM_embedding_chans, layer_id, pretrain)
    else:
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


class ViPT_Block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(ViPT_Block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1,
                                 padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1,
                                 padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1,
                                 padding=0)
        class Fovea(nn.Module):
            def __init__(self, smooth=False):
                super().__init__()
                self.softmax = nn.Softmax(dim=-1)
                self.smooth = smooth
                if smooth:
                    self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

            def forward(self, x):
                '''x: [batch_size, features, k]'''
                b, c, h, w = x.shape
                x = x.contiguous().view(b, c, h * w)
                if self.smooth:
                    mask = self.softmax(x * self.smooth)
                else:
                    mask = self.softmax(x)
                output = mask * x
                output = output.contiguous().view(b, c, h, w)
                return output
        self.fovea = Fovea(smooth=smooth)
        self.patch_norm = nn.LayerNorm(normalized_shape=inplanes)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, rgb_embedding, x_embedding):
        '''
        rgb_embedding: [bs, h, w, c]
        x_embedding: [bs, h, w, c]
        '''
        rgb_embedding_orig = rgb_embedding.clone()
        # calculating residual/updated_x_embedding
        rgb_feat = self.patch_norm(rgb_embedding).permute(0, 3, 1, 2).contiguous()
        x_feat = self.patch_norm(x_embedding).permute(0, 3, 1, 2).contiguous()
        x0 = self.conv0_0(rgb_feat)
        x1 = self.conv0_1(x_feat)
        x0 = self.fovea(x0) + x1
        updated_x_embedding = self.conv1x1(x0).permute(0, 2, 3, 1).contiguous()
        # updated_x_embedding as residual of rgb_embedding
        rgb_embedding = rgb_embedding_orig + updated_x_embedding
        return rgb_embedding, updated_x_embedding


class CMX_Block(nn.Module):
    def __init__(self, SAM_embedding_chanse,layer_id, pretrain):
        assert pretrain is not None
        super(CMX_Block, self).__init__()
        from .sam_naive.modeling.image_encoder import Block
        global_attn_indexes = [2, 5, 8, 11]
        self.x_block = Block(
                args=None,
                dim=SAM_embedding_chanse,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                use_rel_pos=True,
                rel_pos_zero_init=True,
                window_size=14 if layer_id not in global_attn_indexes else 0,
                input_size=(64, 64),
            )
        from .cmx_utils import FeatureRectifyModule as FRM
        self.frm = FRM(dim=SAM_embedding_chanse, reduction=1)
        self.apply(self._init_weights)
        pretrain_x_block = {}
        for n, v in pretrain.items():
            if n.__contains__('image_encoder.blocks.{}.'.format(layer_id)):
                pretrain_x_block[n.split('image_encoder.blocks.{}.'.format(layer_id))[1]] = v
        self.x_block.load_state_dict(pretrain_x_block)

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, rgb_embedding, x_embedding):
        '''
        rgb_embedding: [bs, h, w, c]
        x_embedding: [bs, h, w, c]
        '''
        x_embedding = self.x_block(x_embedding)
        rgb_embedding, x_embedding = rgb_embedding.permute(0,3,1,2), x_embedding.permute(0,3,1,2)
        rgb_embedding, x_embedding = self.frm(rgb_embedding, x_embedding)
        rgb_embedding, x_embedding = rgb_embedding.permute(0,2,3,1), x_embedding.permute(0,2,3,1)
        return rgb_embedding, x_embedding