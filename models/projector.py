import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple, Type
import re

def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    # assert (kernel % 2) == 1, \
    #     'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine=True) -> None:
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels)[None,:, None,None])
            self.bias = nn.Parameter(torch.zeros(num_channels)[None,:, None,None])
        else:
            self.weight = 1
            self.bias = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            args,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size)
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Pre_Callback_Post_Projector(nn.Module):
    def __init__(self, uniform_init, modal_chans, SAM_embedding_chans, proj_type=0, pretrained_state_dict=None):
        uniform_init = uniform_init
        inplane = 4
        super(Pre_Callback_Post_Projector, self).__init__()
        self.SAM_embedding_chans = SAM_embedding_chans
        self.proj_type = proj_type

        if proj_type == 'baseline_a':
            self.zip_rgbx = nn.Identity()
            self.patch_embedding = nn.Conv2d(modal_chans, SAM_embedding_chans, 16, stride=16)

        elif proj_type == 'baseline_b':
            self.zip_rgbx = nn.Sequential(nn.Conv2d(modal_chans, modal_chans * inplane, 3, 1,padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(modal_chans*inplane, 3, 1, 1, padding=1)
                                          )
            self.patch_embedding = nn.Conv2d(3, SAM_embedding_chans, 16, stride=16)
            # init
            if uniform_init:
                nn.init.uniform_(self.zip_rgbx[0].weight)
                nn.init.zeros_(self.zip_rgbx[0].bias)
                nn.init.uniform_(self.zip_rgbx[2].weight)
                nn.init.zeros_(self.zip_rgbx[2].bias)
            if pretrained_state_dict:  # init with pretrained weight. [768,3,16,16]
                patch_embed_weight = pretrained_state_dict['image_encoder.patch_embed.proj.weight']
                patch_embed_bias = pretrained_state_dict['image_encoder.patch_embed.proj.bias']
                self.patch_embedding.load_state_dict({'weight': patch_embed_weight, 'bias': patch_embed_bias})
            else:
                # raise f'requiring pretrained sam to initialize when proj_type={proj_type}'
                print('fail to fine pretrained patch embedding when build projector')

        elif proj_type == 'baseline_c':
            self.zip_rgbx = nn.Conv2d(modal_chans, 3, 1, 1)
            self.patch_embedding = nn.Conv2d(3, SAM_embedding_chans, 16, stride=16)
            # init
            if uniform_init:
                nn.init.uniform_(self.zip_rgbx.weight)
                nn.init.zeros_(self.zip_rgbx.bias)
            if pretrained_state_dict:  # init with pretrained weight. [768,3,16,16]
                patch_embed_weight = pretrained_state_dict['image_encoder.patch_embed.proj.weight']
                patch_embed_bias = pretrained_state_dict['image_encoder.patch_embed.proj.bias']
                self.patch_embedding.load_state_dict({'weight': patch_embed_weight, 'bias': patch_embed_bias})
            else:
                # raise f'requiring pretrained sam to initialize when proj_type={proj_type}'
                print('fail to fine pretrained patch embedding when build projector')

        elif proj_type == 'baseline_d':
            self.zip_rgbx = nn.Conv2d(modal_chans, 3, 1, 1)
            # init
            if uniform_init:
                nn.init.uniform_(self.zip_rgbx.weight)
                nn.init.zeros_(self.zip_rgbx.bias)

        elif proj_type =='simcmf':
            self.moe = nn.Sequential(nn.Conv2d(modal_chans, modal_chans*inplane, 3, 1, padding=1),
                                     LayerNorm2d(modal_chans*inplane),
                                     nn.ReLU(),
                                     nn.Conv2d(modal_chans * inplane, modal_chans, 3, 1, padding=1),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )
            # preserve old from rgbx
            self.zip_rgbx = nn.Conv2d(modal_chans, 3, 1, 1)
            if uniform_init:
                nn.init.uniform_(self.zip_rgbx.weight)
                nn.init.zeros_(self.zip_rgbx.bias)
            # learn novel from rgbx
            self.group_patch_embedding = nn.ModuleList()
            for _ in range(modal_chans):
                module = nn.Conv2d(1, SAM_embedding_chans, 16, stride=16)
                # init
                if pretrained_state_dict:  # init with pretrained weight. [768,3,16,16]
                    patch_embed_weight = pretrained_state_dict['image_encoder.patch_embed.proj.weight'].sum(1,keepdim=True)
                    patch_embed_bias = pretrained_state_dict['image_encoder.patch_embed.proj.bias']
                    module.load_state_dict({'weight': patch_embed_weight, 'bias': patch_embed_bias})
                else:
                    # raise f'requiring pretrained sam to initialize when proj_type={proj_type}'
                    print('fail to fine pretrained patch embedding when build projector')
                self.group_patch_embedding.append(module)
            self.fc = nn.Linear(modal_chans, 1, bias=False)
            nn.init.zeros_(self.fc.weight)

        elif proj_type.__contains__('preconv'):
            conf = proj_type.split('preconv')[1]
            kernel_size = int(conf.split('k')[1].split('n')[0])
            num_layer = int(conf.split('n')[1].split('d')[0])
            dim = int(conf.split('d')[1])
            if num_layer == 1:
                zip_proj = [nn.Conv2d(modal_chans, 3, kernel_size, 1, padding=kernel_size // 2)]
            else:
                zip_proj = [nn.Conv2d(modal_chans, dim, kernel_size, 1, padding=kernel_size//2)]
                for i in range(num_layer-2):
                    zip_proj.append(nn.Conv2d(dim, dim, kernel_size, 1, padding=kernel_size//2))
                    zip_proj.append(nn.ReLU())
                zip_proj.append(nn.Conv2d(dim, 3, 1, 1))
            self.zip_proj = nn.Sequential(*zip_proj)

        elif proj_type.__contains__('preattn'):
            conf = proj_type.split('preattn')[1]
            patch_size = int(conf.split('p')[1].split('n')[0])
            num_layer = int(conf.split('n')[1].split('d')[0])
            dim = int(conf.split('d')[1])
            self.pre_attn_patch_size = patch_size
            zip_proj = [nn.Conv2d(modal_chans, dim, patch_size, patch_size)]
            for i in range(num_layer):
                zip_proj.append(Block(None, dim, num_heads=8))
            zip_proj.append(nn.Linear(dim, patch_size*patch_size*3))
            self.zip_proj = nn.Sequential(*zip_proj)

        elif proj_type == 'zeroshot':
            pass

        else:
            raise NotImplementedError


    def forward(self, input, func):
        '''
        args:
            input: [bs, 3+x_dim, H, W]
            func: SAM_patch_embedding_layer
        '''


        if self.proj_type in ['baseline_a', 'baseline_b', 'baseline_c']:
            output = self.zip_rgbx(input)
            output = self.patch_embedding(output)
            output = output.permute(0,2,3,1)
            return output, None

        elif self.proj_type == 'baseline_d':
            output = self.zip_rgbx(input)
            output = func(output)
            return output, None


        elif self.proj_type == 'simcmf':
            weights = torch.sigmoid(self.moe(input))
            up_branch, down_branch = input * weights, input * (1-weights)
            # old
            old_embedding = func(self.zip_rgbx(up_branch))
            # novel
            bs, num_groups, h, w = input.shape
            rgbx = down_branch.unsqueeze(2)  # [bs, num_channel, 1, h, w]
            novel_embedding = torch.stack([self.group_patch_embedding[i](rgbx[:, i]) for i in range(num_groups)])
            novel_embedding = novel_embedding.permute(1, 3, 4, 2, 0)  # [bs, h, w, C, N]
            novel_embedding = self.fc(novel_embedding)[..., 0]
            output = old_embedding + novel_embedding
            # print(f'weights is {weights}')
            # print(f'old norm vs novel norm:{old_embedding.norm()} vs {novel_embedding.norm()}')
            return output, None

        elif self.proj_type.__contains__('preconv'):
            rgb_embedding = self.zip_proj(input)
            rgb_embedding = func(rgb_embedding)
            return rgb_embedding, None

        elif self.proj_type.__contains__('preattn'):
            b, _, h, w = input.shape
            rgb_embedding = self.zip_proj[0](input).permute(0,2,3,1)
            rgb_embedding = self.zip_proj[1:](rgb_embedding)
            b, c = rgb_embedding.shape[:2]
            rgb_embedding = rgb_embedding.reshape(b, h//self.pre_attn_patch_size, w//self.pre_attn_patch_size, self.pre_attn_patch_size , self.pre_attn_patch_size, 3)
            rgb_embedding = rgb_embedding.reshape(b, h, w, 3).permute(0,3,1,2)
            rgb_embedding = func(rgb_embedding)
            return rgb_embedding, None

        elif self.proj_type == 'zeroshot':
            if input.shape[1]<3:
                input = F.pad(input,(0,0,0,0,0,3), mode='replicate')
            rgb_embedding = func(input[:,:3])
            return rgb_embedding, None


        else:
            raise NotImplementedError






