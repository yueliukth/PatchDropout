# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# MAE: https://github.com/facebookresearch/mae

from copy import deepcopy
from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.vision_transformer import checkpoint_filter_fn, _init_vit_weights
from timm.models.swin_transformer import window_reverse, window_partition
from timm.models.layers import PatchEmbed, Mlp, DropPath, to_2tuple, trunc_normal_

import helper

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'swin_base_patch4_window12_384': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_base_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',
    ),

    'swin_large_patch4_window12_384': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'swin_large_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',
    ),

    'swin_small_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    ),

    'swin_tiny_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    ),

    'swin_base_patch4_window12_384_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841),

    'swin_base_patch4_window7_224_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        num_classes=21841),

    'swin_large_patch4_window12_384_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841),

    'swin_large_patch4_window7_224_in22k': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
        num_classes=21841),

}


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C) # B, 8n, 8n, 128

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None, ids_keep:Optional[torch.Tensor] = None, batch_size:Optional[torch.Tensor] = None, rank:Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q/k/v: 64*B, num_heads, num_tokens, dim/num_heads
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # 64*B, 4, 4, 4

        relative_position_index = self.relative_position_index  # 49, 49
        relative_position_index = torch.cat(batch_size*[relative_position_index[None, :, :]])  # B, 49, 49

        relative_position_index_masked = torch.gather(relative_position_index, dim=1,
                                                      index=ids_keep.unsqueeze(-1).repeat(1, 1, self.window_size[0]*self.window_size[1]))  # B, num_after_sampling, 49
        relative_position_index_masked = torch.transpose(relative_position_index_masked, 1, 2)  # B, 49, num_after_sampling
        relative_position_index_masked = torch.gather(relative_position_index_masked, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, N))  # B. num_after_sampling, num_after_sampling
        relative_position_index_masked = torch.transpose(relative_position_index_masked, 1, 2)  # B. num_after_sampling, num_after_sampling

        relative_position_bias = torch.index_select(self.relative_position_bias_table, 0, relative_position_index_masked.flatten()).view(batch_size, N, N, -1)  # B, num_after_sampling, num_after_sampling, 4
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()  # B, 4, num_after_sampling, num_after_sampling
        relative_position_bias = relative_position_bias.repeat(int(B_/batch_size),1,1,1)
        attn = attn + relative_position_bias  # 64*B, 4, num_after_sampling, num_after_sampling  + 64*B, 4, num_after_sampling, num_after_sampling

        if mask is not None:
            nW = mask.shape[0]  # 64, 49, 49
            mask = mask.repeat(int(attn.size()[0]/nW), 1, 1)  # 64*B, 49, 49
            ids_keep = ids_keep.repeat(int(B_/batch_size), 1)  # 64*B, num_after_sampling*num_after_sampling
            mask = torch.gather(mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.window_size[0]*self.window_size[1]))  # 64*B, num_after_sampling, 49
            mask = torch.transpose(mask, 1, 2)  # 64*B, 49, num_after_sampling
            mask = torch.gather(mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, N))  # 64*B. num_after_sampling, num_after_sampling
            mask = torch.transpose(mask, 1, 2)  # 64*B. num_after_sampling, num_after_sampling
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).view(B_ // nW, nW, 1, N, N)
            attn = attn.view(-1, self.num_heads, N, N)  #64*B, number_heads, 4, 4
            attn = self.softmax(attn)  #64*B, number_heads, 4, 4
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 64*B, num_after_sampling, 128
        x = self.proj(x)  # 64*B, num_after_sampling, 128
        x = self.proj_drop(x)  # 64*B, num_after_sampling, 128

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # 64, 49, 49
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, input):
        rank, x, ids_keep = input
        new_window_size = int(math.sqrt(ids_keep.size()[1]))

        # H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        H = W = int(math.sqrt(L))

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # (B, 56, 56, 128), or (B, n*8, n*8, 128) while n is the new window size

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, new_window_size)  # (64*B, 7, 7, 128) or (64*B, n, n, 128)
        x_windows = x_windows.view(-1, new_window_size * new_window_size, C)  # (64*B, 49, 128) or (64*B, n*n, 128)

        # W-MSA/SW-MSA
        # self.attn_mask is None when self.shift_size==0
        attn_windows = self.attn(x_windows, mask=self.attn_mask, ids_keep=ids_keep, batch_size=B, rank=rank)  # 64*B, n*n, 128

        # merge windows
        attn_windows = attn_windows.view(-1, new_window_size, new_window_size, C)  # 64*B, n, n, 128
        shifted_x = window_reverse(attn_windows, new_window_size, H, W)  # B, 8n, 8n, C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, -1, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return rank, x, ids_keep


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, input):
        rank, x, ids_keep = input

        for blk in self.blocks:
            rank, x, ids_keep = blk((rank, x, ids_keep))  # B, 8n*8n, 128

        if self.downsample is not None:
            x = self.downsample(x)
        return rank, x, ids_keep


class SwinTransformer(timm.models.swin_transformer.SwinTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(SwinTransformer, self).__init__(**kwargs)
        self.patch_size = kwargs['patch_size']
        # swin_base_patch4_window7_224_in22k -- from kwargs:
        # patch_size=4,
        # window_size=7,
        # embed_dim=128,
        # depths=(2, 2, 18, 2),
        # num_heads=(4, 8, 16, 32)

        depths = kwargs['depths']
        num_heads = kwargs['num_heads']
        window_size = kwargs['window_size']
        self.window_size = window_size

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, 0, sum(depths))]  # stochastic depth decay rule

        self.starting_input_resolution = (self.patch_grid[0] // (2 ** 0), self.patch_grid[1] // (2 ** 0))

        # build layers
        layers = []
        for i_layer in range(self.num_layers):
            layers += [BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(self.patch_grid[0] // (2 ** i_layer), self.patch_grid[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop=0,
                attn_drop=0,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # [0.0] * depth
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False)
            ]
        self.layers = nn.Sequential(*layers)

    def random_masking(self, rank, x, window_size_after_masking, batch_size):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        Returns:
            x_masked: masked representations
            mask: None
            ids_keep: ids used to sample tokens
        """
        device = torch.device("cuda:{}".format(rank))
        N, L, D = x.shape  # batch, length, dim

        indices = torch.range(0, self.window_size*self.window_size-1, device=device).view(-1, self.window_size, self.window_size).repeat(batch_size, 1, 1)  # B, 7, 7

        noise_x = torch.rand(batch_size, self.window_size, device=device)  # noise in [0, 1]
        noise_y = torch.rand(batch_size, self.window_size, device=device)  # noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle_x = torch.argsort(noise_x, dim=1)  # ascend: small is keep, large is remove
        # Keep the first subset
        ids_keep_x = ids_shuffle_x[:, :window_size_after_masking]
        ids_keep_x = torch.sort(ids_keep_x, dim=1).values  # not to shuffle tokens

        # Sort noise for each sample
        ids_shuffle_y = torch.argsort(noise_y, dim=1)  # ascend: small is keep, large is remove
        # Keep the first subset
        ids_keep_y = ids_shuffle_y[:, :window_size_after_masking]
        ids_keep_y = torch.sort(ids_keep_y, dim=1).values  # not to shuffle tokens

        indices = torch.gather(indices, dim=1, index=ids_keep_x.unsqueeze(-1).repeat(1, 1, self.window_size))  # B, number_tokens_left, 7
        indices = torch.swapaxes(indices, 1, 2)  # B, 7, number_tokens_left
        indices = torch.gather(indices, dim=1, index=ids_keep_y.unsqueeze(-1).repeat(1, 1, window_size_after_masking))  # B, number_tokens_left, number_tokens_left
        indices = torch.swapaxes(indices, 1, 2)   # B, number_tokens_left, number_tokens_left

        ids_keep = torch.flatten(indices, start_dim=1).type(torch.int64)  # B, number_tokens_left*number_tokens_left
        x_masked = torch.gather(x, dim=1, index=ids_keep.repeat(int(N/batch_size),1).unsqueeze(-1).repeat(1, 1, D))
        mask = None
        return x_masked, mask, ids_keep

    def forward_features(self, rank, x, keep_rate):
        x = self.patch_embed(x)  # B, 56*56, 128 --> # B, 3136, 128

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)  # B, 56*56, 128 --> # B, 3136, 128

        # NEW: added random masking
        H, W = self.starting_input_resolution
        B, L, C = x.shape  # x: B, 56*56, 128
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)  # B, 56, 56, 128

        # partition windows
        x_windows = window_partition(x, self.window_size)  # 64*B, 7, 7, 128
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # 64*B, 49, 128

        window_size_after_masking = int(keep_rate*self.window_size)
        x_windows, mask, ids_keep = self.random_masking(rank, x_windows, window_size_after_masking, batch_size=B)

        H = W = int(math.sqrt(x_windows.size()[0]/B))*window_size_after_masking
        x = window_reverse(x_windows, window_size_after_masking, H, W)  # B, 8n, 8n, C
        x = x.view(B, -1, C)   # B, 8n*8n, C

        rank, x, ids_keep = self.layers((rank, x, ids_keep))  # B, n*n, 1024
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B, 1024, 1
        x = torch.flatten(x, 1)  # B, 1024
        return x

    def forward(self, rank, x, keep_rate, dynamic_keep_rate):
        if dynamic_keep_rate:
            raise NotImplementedError(f"dynamic_keep_rate has not been implemented with SWINs")
        x = self.forward_features(rank, x, keep_rate)
        x = self.head(x)  # B, num_classes
        return x


def _create_swin_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        SwinTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model


@register_model
def swin_base_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer('swin_base_patch4_window12_384', pretrained=pretrained, **model_kwargs)


@register_model
def swin_base_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer('swin_base_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_large_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-L @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer('swin_large_patch4_window12_384', pretrained=pretrained, **model_kwargs)


@register_model
def swin_large_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-L @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer('swin_large_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_small_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer('swin_small_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_tiny_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer('swin_tiny_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_base_patch4_window12_384_in22k(pretrained=False, **kwargs):
    """ Swin-B @ 384x384, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer('swin_base_patch4_window12_384_in22k', pretrained=pretrained, **model_kwargs)


@register_model
def swin_base_patch4_window7_224_in22k(pretrained=False, **kwargs):
    """ Swin-B @ 224x224, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer('swin_base_patch4_window7_224_in22k', pretrained=pretrained, **model_kwargs)


@register_model
def swin_large_patch4_window12_384_in22k(pretrained=False, **kwargs):
    """ Swin-L @ 384x384, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer('swin_large_patch4_window12_384_in22k', pretrained=pretrained, **model_kwargs)


@register_model
def swin_large_patch4_window7_224_in22k(pretrained=False, **kwargs):
    """ Swin-L @ 224x224, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer('swin_large_patch4_window7_224_in22k', pretrained=pretrained, **model_kwargs)

