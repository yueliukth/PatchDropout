# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# MAE: https://github.com/facebookresearch/mae

import random
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from functools import partial
import logging
import torch
import torch.nn as nn
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg, named_apply
from timm.models.vision_transformer import checkpoint_filter_fn
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
import helper
from patchdropout import PatchDropout

_logger = logging.getLogger(__name__)

class Attention(timm.models.vision_transformer.Attention):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # eg, torch.Size([32, 12, 10, 197])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)  # torch.Size([32, 10, 768])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(timm.models.vision_transformer.Block):
    def __init__(self, **kwargs):
        self.block_index = kwargs['block_index']
        self.block_index_max = kwargs['block_index_max']
        kwargs.pop('block_index', None)
        kwargs.pop('block_index_max', None)
        super(Block, self).__init__(**kwargs)
        self.attn = Attention(dim=kwargs['dim'], num_heads=kwargs['num_heads'], qkv_bias=kwargs['qkv_bias'],
                              attn_drop=kwargs['attn_drop'], proj_drop=kwargs['drop'])

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with other support
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.patch_size = kwargs['patch_size']
        self.num_patches = self.patch_embed.num_patches
        self.depth = kwargs['depth']

        block_depth = self.depth
        block_index_max = self.depth-1
        dpr = [x.item() for x in torch.linspace(0, 0, block_depth)]  # stochastic depth decay rule
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.blocks = nn.Sequential(*[
            Block(
                dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=4, qkv_bias=True, drop=0.,
                attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, block_index=i,
                block_index_max=block_index_max)
            for i in range(self.depth)])

        self.init_weights(mode='')

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_features(self, x, keep_rate, random_keep_rate):
        N = x.size()[0]  # batch, length, dim

        # Get patch embeddings without cls tokens
        patch_embeddings = self.patch_embed(x) + self.pos_embed[:, self.num_tokens:, :]
        
        # Prepare cls token
        cls_token = self.cls_token.expand(N, -1, -1) + self.pos_embed[:, :1, :]  # cls_tokens impl from Phil Wang

        # Cat CLS token
        x = torch.cat((cls_token, patch_embeddings), dim=1)

        # Random keep rate if random_keep_rate is True
        if random_keep_rate:
            start_mask_ratio = 0.0
            end_mask_ratio = 0.5
            if random.random() < 0.5:
                random_mask_ratio = random.randint(int(start_mask_ratio*20), int(end_mask_ratio*20))/20
                keep_rate = 1 - random_mask_ratio
            else:
                keep_rate = 1

        # PatchDropout
        x = PatchDropout(keep_rate)(x, force_drop=True)
        
        x = self.pos_drop(x)  # always drop (if planned) before blocks
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, rank, x, keep_rate, random_keep_rate):
        x = self.forward_features(x, keep_rate, random_keep_rate)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        return x

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
    ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    'vit_huge_patch14_224': _cfg(url=''),
    'vit_giant_patch14_224': _cfg(url=''),
    'vit_gigantic_patch14_224': _cfg(url=''),

    'vit_base2_patch32_256': _cfg(url='', input_size=(3, 256, 256), crop_pct=0.95),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_cifar100': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch8_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),
    'vit_huge_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch16_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_224_sam': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_224_sam': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'vit_small_patch16_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch8_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch16_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch8_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base2_patch32_256(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32)
    # FIXME experiment
    """
    model_kwargs = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, **kwargs)
    model = _create_vision_transformer('vit_base2_patch32_256', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    """ ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    """ ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_gigantic_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_sam(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    # NOTE original SAM weights release worked with representation_size=768
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_sam', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_sam(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    # NOTE original SAM weights release worked with representation_size=768
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_sam', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_dino(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch8_224_dino(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch8_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_dino(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) /w DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224_dino(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        'deit_small_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'deit_base_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
    return model
