import random
import itertools
from copy import deepcopy
from functools import partial
import torch
from torchvision import models as torchvision_models
import torch.nn as nn
from torch.cuda.amp import autocast
from einops.layers.torch import Rearrange
import timm
import timm.models.swin_transformer as swin_transformers
from timm.models.swin_transformer import _create_swin_transformer

import timm_vit
import timm_swin
import helper


# ============ Timm models ... ============
class TimmModels(nn.Module):
    def __init__(self, backbone_option, **kwargs):
        super().__init__()
        if 'vit' in backbone_option or 'deit' in backbone_option:
            self.model = timm_vit.__dict__[backbone_option](**kwargs)
        elif 'swin' in backbone_option:
            self.model = timm_swin.__dict__[backbone_option](**kwargs)