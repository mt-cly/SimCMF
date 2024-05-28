import torch
import torch.nn as nn
from functools import partial
import torch.utils.checkpoint as cp
# from ops.modules import MSDeformAttn
from .cmx_utils import FeatureFusionModule as FFM


def get_final_block(SAM_embedding_chans, proj_type):
    if proj_type == 'cmx':
        return CMX_Final_Block(SAM_embedding_chans)
    else:
        return Identity_Block()

class Identity_Block(nn.Module):
    def __init__(self):
        super(Identity_Block, self).__init__()

    def forward(self, ms_rgb_embedding, ms_x_embedding):
        return ms_rgb_embedding[-1]

class CMX_Final_Block(nn.Module):
    def __init__(self, SAM_embedding_chans):
        super(CMX_Final_Block, self).__init__()
        norm_fuse = nn.BatchNorm2d
        embed_dims = [SAM_embedding_chans, SAM_embedding_chans, SAM_embedding_chans, SAM_embedding_chans]
        num_heads = [12, 12, 12, 12]
        self.FFMs = nn.ModuleList([
            FFM(dim=embed_dims[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
            FFM(dim=embed_dims[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
            FFM(dim=embed_dims[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
            FFM(dim=embed_dims[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])
        self.linear = nn.Linear(4, 1)
        self.apply(self._init_weights)

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

    def forward(self, ms_rgb_embedding, ms_x_embedding):
        fused_embs = []
        for rgb_emb, x_emb, ffm in zip(ms_rgb_embedding, ms_x_embedding, self.FFMs):
            emb = ffm(rgb_emb.permute(0,3,1,2), x_emb.permute(0,3,1,2))
            fused_embs.append(emb.permute(0,2,3,1))
        fused_embs = torch.stack(fused_embs, dim=-1)
        fused_embs = self.linear(fused_embs)[..., 0]

        return fused_embs