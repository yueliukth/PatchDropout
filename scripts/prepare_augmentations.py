import ast
import random
from PIL import Image
from PIL import ImageFilter, ImageOps
import torch
from torchvision import transforms
from RandAugment import RandAugment

import helper

class PublicDataAugmentation(object):
    # Adopted from the original DINO implementation
    # Removed global_size and local_size
    def __init__(self, dataset_params):
        dataset_name = dataset_params['dataset_name']
        cropping_ratio = float(256/224)
        full_size = int(dataset_params['resolution']*cropping_ratio)
        global_size = int(dataset_params['resolution'])

        # Define the normalization
        normalize_mean = ast.literal_eval(str(dataset_params['dataset_choice'][dataset_name]['normalize_mean']))
        normalize_std = ast.literal_eval(str(dataset_params['dataset_choice'][dataset_name]['normalize_std']))
        self.normalize = transforms.Compose([transforms.Normalize(normalize_mean, normalize_std),])

        # Define transforms for training (transforms_aug) and for validation (transforms_plain)
        self.transforms_plain = transforms.Compose([
            transforms.Resize(full_size, interpolation=3),
            transforms.CenterCrop(global_size),  # Center cropping
            transforms.ToTensor()])

        self.transforms_aug = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)), # default imagenet scale/ratio range
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(*(float(0.4),) * 3),
            transforms.ToTensor()])

    def __call__(self, image):
        crops = []
        crops.append(self.transforms_aug(image))
        return crops


class MammoaiDataAugmentation(object):
    # Adopted from the original DINO implementation
    # Removed global_size and local_size
    def __init__(self, dataset_params):
        dataset_name = dataset_params['dataset_name']
        # cropping_ratio = float(256/224)
        # full_size = int(dataset_params['resolution']*cropping_ratio)
        global_size = int(dataset_params['resolution'])

        normalize_mean = ast.literal_eval(str(dataset_params['dataset_choice'][dataset_name]['normalize_mean']))
        normalize_std = ast.literal_eval(str(dataset_params['dataset_choice'][dataset_name]['normalize_std']))
        self.normalize = transforms.Compose([transforms.Normalize(normalize_mean, normalize_std),])

        self.transforms_plain = transforms.Compose([
            transforms.Resize((int(global_size), int(global_size))),
            transforms.ToTensor()])

        self.transforms_aug = transforms.Compose([
                transforms.Resize((int(global_size*1.125), int(global_size*1.125,))),
                transforms.RandomResizedCrop(global_size, scale=(1, 1), interpolation=3),
                transforms.ColorJitter(brightness=0.1, contrast=0.5, saturation=0, hue=0),
                transforms.ToTensor()])

    def __call__(self, image):
        crops = []
        crops.append(self.transforms_aug(image))
        return crops

