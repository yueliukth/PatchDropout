import os
import json
import shutil
import random
import tarfile
import numpy as np
import pandas as pd
import cv2
from glob import glob
from PIL import Image
from PIL import ImageFilter, ImageOps
from operator import itemgetter
from typing import Iterator, List, Optional, Union

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
import torch.distributed as dist

from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import Sampler

import helper


class SimilarLengthBalancedClassSampler(Sampler):
    def __init__(self, dataset, similar_length_column, if_similar_length=True, if_balanced_class=True):
        self.dataset = dataset
        self.label_csv = dataset.label_csv
        self.similar_length_column = similar_length_column
        self.if_similar_length = if_similar_length
        self.if_balanced_class = if_balanced_class

        if if_balanced_class:
            self.original_neg_to_pos = int(dataset.neg_num / dataset.pos_num)
            self.pos_num = dataset.pos_num
            self.len = int(self.pos_num/0.5)
        else:
            self.len = self.label_csv.shape[0]

    def __iter__(self):
        if helper.is_main_process():
            print('Iterating SimilarLengthBalancedClassSampler...')

        label_csv = self.label_csv.copy()
        if self.if_balanced_class:
            # get all pos images and a random sample of neg images (same number as pos)
            idxs = []
            # hospital_a data
            for manu in list(set(label_csv['dicom_manufacturer'].tolist())):
                manu_label_csv = label_csv[(label_csv['dicom_manufacturer'] == manu) & (label_csv['if_hospitala'] == 1)]
                manu_pos_idxs = manu_label_csv[manu_label_csv['label'] == 1].index.tolist()
                manu_neg_idxs = manu_label_csv[manu_label_csv['label'] == 0].index.tolist()
                manu_pos = np.random.choice(np.array(manu_pos_idxs), len(manu_pos_idxs), replace=False)
                try:
                    manu_neg = np.random.choice(np.array(manu_neg_idxs), len(manu_pos_idxs), replace=False)
                except:
                    manu_neg = np.random.choice(np.array(manu_neg_idxs), len(manu_pos_idxs), replace=True)
                manu_idxs = np.hstack((manu_pos, manu_neg)).tolist()
                idxs.extend(manu_idxs)
            # non hospital_a data
            for manu in list(set(label_csv['dicom_manufacturer'].tolist())):
                manu_label_csv = label_csv[(label_csv['dicom_manufacturer'] == manu) & (label_csv['if_hospitala'] != 1)]
                manu_pos_idxs = manu_label_csv[manu_label_csv['label'] == 1].index.tolist()
                manu_neg_idxs = manu_label_csv[manu_label_csv['label'] == 0].index.tolist()
                manu_pos = np.random.choice(np.array(manu_pos_idxs), len(manu_pos_idxs), replace=False)
                try:
                    manu_neg = np.random.choice(np.array(manu_neg_idxs), len(manu_pos_idxs), replace=False)
                except:
                    manu_neg = np.random.choice(np.array(manu_neg_idxs), len(manu_pos_idxs), replace=True)
                manu_idxs = np.hstack((manu_pos, manu_neg)).tolist()
                idxs.extend(manu_idxs)

            random.shuffle(idxs)
            if helper.is_main_process():
                print('Manufacturer sampler...')
                print('Number of samples: ', len(idxs), len(set(idxs)))
                new_df = label_csv.loc[idxs, :]
                for manu in list(set(new_df['dicom_manufacturer'].tolist())):
                    manu_hospital_a_new_df = new_df[(new_df['dicom_manufacturer'] == manu) & (new_df['if_hospitala'] == 1)]
                    manu_nonhospitala_new_df = new_df[(new_df['dicom_manufacturer'] == manu) & (new_df['if_hospitala'] != 1)]
                    manu_hospitala_pos_num = manu_hospital_a_new_df[manu_hospitala_new_df['label'] == 1].shape[0]
                    manu_hospitala_neg_num = manu_hospitala_new_df[manu_hospitala_new_df['label'] == 0].shape[0]
                    manu_nonhospitala_pos_num = manu_nonhospitala_new_df[manu_nonhospitala_new_df['label'] == 1].shape[0]
                    manu_nonhospitala_neg_num = manu_nonhospitala_new_df[manu_nonhospitala_new_df['label'] == 0].shape[0]
                    print(f'Sampled - manu {manu}: hospitala pos/neg {manu_hospitala_pos_num} / {manu_hospitala_neg_num}')
                    print(f'Sampled - manu {manu}: nonhospitala pos/neg {manu_nonhospitala_pos_num} / {manu_nonhospitala_neg_num}')
        else:
            idxs = label_csv.index.tolist()
            random.shuffle(idxs)

        if self.if_similar_length:
            # sort the indices by similar_length_column (note that the order of ties is randomized)
            similar_length_dict = zip(idxs, label_csv.loc[idxs, self.similar_length_column].tolist())
            idxs = sorted(similar_length_dict, key=lambda x: x[1], reverse=False)
            idxs = [x[0] for x in idxs]

        if self.if_balanced_class:
            # get a list of indices of dataset length
            idxs = idxs[:self.len]
        return iter(idxs)

    def __len__(self):
        return self.len


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    From: https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
            self,
            sampler,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class CSAWDataset(Dataset):
    def __init__(self, img_folder, csv_path, official_split, transforms=None, normalize=None,
                 input_type_name=None, patch_size=16, global_size=224):
        """
        Args
        """
        if helper.is_main_process():
            print('Initializing CSAWDataset...')

        self.img_folder = img_folder
        self.official_split = official_split
        self.input_type_name = input_type_name

        self.transforms = transforms
        self.normalize = normalize

        self.patch_size = patch_size
        self.global_size = global_size

        # Get csv file and get the rows with images that exist in img_folder
        csv = pd.read_csv(csv_path, sep=';')
        csv['basename_png'] = csv['basename'].str[:-3]+'png'
        if helper.is_main_process():
            print(f"Number of images written in csv {csv_path}: {len(set(csv['basename_png']))}")
        png_list = glob(os.path.join(img_folder, '*.png'))
        png_list = [os.path.basename(i) for i in png_list]
        csv = csv[csv['basename_png'].isin(png_list)].reset_index(drop=True)
        if helper.is_main_process():
            print(f"Number of images written in csv {csv_path} and existing in path {img_folder}: {len(set(csv['basename_png']))}")

        # Get label_csv according to the needed data split and the task type (cancer or risk)
        split_column_val = 'split_risk'
        patient_selected_column = 'risk_prediction_patient_selected'
        label_column_image_level = 'breast_level_risk'
        if 'train' in official_split:
            self.label_csv = csv[(csv['split'] == 'train') & (csv[patient_selected_column] == 1)].reset_index(drop=True)

        elif 'val' in official_split:  # make sure we have class balance in validation set
            self.label_csv = csv[(csv[split_column_val] == 'val_selected') & (csv[patient_selected_column] == 1)].reset_index(drop=True)
        elif 'test' in official_split:  # note here we do inference on all val+test images (to double check whether
            # the val performance is correctly recorded)
            self.label_csv = csv[((csv['split'] == 'test') | (csv[split_column_val] == 'val_selected')) & (csv[patient_selected_column] == 1)].reset_index(drop=True)

        # Get label_csv according to task type (whether inputs accumulate vies or not)
        self.label_csv['label'] = self.label_csv[label_column_image_level]
        self.num_fgnd_column = 'num_fgnd'

        # Print number of pos/neg samples
        self.pos_num = self.label_csv[self.label_csv['label'] == 1].shape[0]
        self.neg_num = self.label_csv[self.label_csv['label'] == 0].shape[0]
        if helper.is_main_process():
            print(f'Number of {official_split} samples - pos/neg {self.pos_num} / {self.neg_num}')

    def get_image_only(self, idx, img_path, flip_p):
        # Read image array with opencv
        img_array = cv2.imread(img_path)
        img = Image.fromarray(img_array, mode='RGB')

        if 'train' in self.official_split:
            # Random rotation on BOTH img and breast mask
            rand_angle = random.randint(-10, 10)
            img = torchvision.transforms.functional.rotate(img, rand_angle)

            # Random flipping on img
            if flip_p > 0.5:
                img = torchvision.transforms.functional.hflip(img)

        # Transformations
        img = self.transforms(img)

        # Normalise the inputs
        img = self.normalize(img)

        # Get image label 1, or 0
        lab = self.label_csv.loc[idx, 'label']
        exam_note = self.label_csv.loc[idx, 'exam_note']
        basename = os.path.basename(img_path)
        return img, lab, basename, exam_note

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.label_csv.loc[idx, 'basename_png'])
        return self.get_image_only(idx, img_path, flip_p=random.random())

    def __len__(self):
        return len(self.label_csv)


class GetCSAWDatasets():
    def __init__(self, dataset_params, normalize, patch_size, input_type_name):
        self.dataset_params = dataset_params
        self.normalize = normalize
        self.patch_size = patch_size
        self.input_type_name = input_type_name
        self.img_folder = os.path.join(dataset_params['data_folder'], "pngs_native")
        if helper.is_main_process():
            print(f'Data folder is at {self.img_folder}')
        self.global_size = int(dataset_params['resolution'])

        dataset_name = dataset_params['dataset_name']
        self.csv_path = os.path.join(dataset_params['data_folder'],
                                     dataset_params['dataset_choice'][dataset_name]['csv_path'])

    def get_datasets(self, official_split, transforms):
        dataset = CSAWDataset(img_folder=self.img_folder, csv_path=self.csv_path,
                              official_split=official_split,
                              transforms=transforms,
                              normalize=self.normalize,
                              input_type_name=self.input_type_name,
                              patch_size=self.patch_size, global_size=self.global_size
                              )
        
        if helper.is_main_process():
            print(f"There are {len(dataset)} samples in {official_split} split, on each rank. ")
        return dataset


class ImageFolderReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ImageFolderReturnIndexDataset, self).__getitem__(idx)
        return img, lab, idx

class ReturnIndexDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.subset)


class GetPublicDatasets():
    def __init__(self, dataset_params, transforms_aug, transforms_plain, normalize):
        self.dataset_params = dataset_params
        self.transforms_aug = transforms_aug
        self.transforms_plain = transforms_plain
        self.normalize = normalize

    def get_datasets(self, official_split):
        if self.dataset_params['dataset_name'] == 'CIFAR100':
            # Train: 50,000 images
            # Test: 10,000 images

            if official_split == 'train/' or official_split == 'val/':
                # Note that CIFAR dataset doesn't have its official validation set, so here we create our own
                original_train_dataset = datasets.CIFAR100(
                    root=self.dataset_params['data_folder'],
                    train=True,
                    download=False,
                    transform=None)

                num_train = len(original_train_dataset)
                valid_size = 0.02
                split = int(np.floor(valid_size * num_train))
                train_set, valid_set = torch.utils.data.random_split(original_train_dataset, [num_train-split, split],
                                                                     generator=torch.Generator().manual_seed(42))

                train_set = ReturnIndexDataset(train_set, transform=torchvision.transforms.Compose([self.transforms_aug, self.normalize]))
                valid_set = ReturnIndexDataset(valid_set, transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))

                if helper.is_main_process():
                    print(f"There are {len(train_set)} samples in train split, on each rank. ")
                    print(f"There are {len(valid_set)} samples in val split, on each rank. ")
                return train_set, valid_set
            else:
                dataset = datasets.CIFAR100(
                    root=self.dataset_params['data_folder'],
                    train=False,
                    download=False,
                    transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))
                dataset = ReturnIndexDataset(dataset, transform=None)
                return dataset
        elif self.dataset_params['dataset_name'] == 'Places365':
            # The Places365 dataset is a scene recognition dataset.
            # It is composed of 10 million images comprising 434 scene classes.
            # There are two versions of the dataset:
            # Places365-Standard with 1.8 million train and 36000 validation images from K=365 scene classes,
            # and Places365-Challenge-2016, in which the size of the training set is increased up to 6.2 million extra images,
            # including 69 new scene classes (leading to a total of 8 million train images from 434 scene classes).
            if official_split == 'train/' or official_split == 'val/':
                # Note that Places365 dataset doesn't have its official validation set, so here we create our own
                # There are 1785426 samples in train split, on each rank.
                # There are 18034 samples in val split, on each rank.

                original_train_dataset = datasets.Places365(root=os.path.join(self.dataset_params['data_folder'], 'places365/'),
                                                            split='train-standard',
                                                            small=True,
                                                            download=False,
                                                            transform=None)

                num_train = len(original_train_dataset)
                valid_size = 0.01
                split = int(np.floor(valid_size * num_train))
                train_set, valid_set = torch.utils.data.random_split(original_train_dataset, [num_train-split, split],
                                                                     generator=torch.Generator().manual_seed(42))
                train_set = ReturnIndexDataset(train_set, transform=torchvision.transforms.Compose([self.transforms_aug, self.normalize]))
                valid_set = ReturnIndexDataset(valid_set, transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))

                if helper.is_main_process():
                    print(f"There are {len(train_set)} samples in train split, on each rank. ")
                    print(f"There are {len(valid_set)} samples in val split, on each rank. ")
                return train_set, valid_set
            else:
                dataset = datasets.Places365(root=self.dataset_params['data_folder'],
                                             split='val',
                                             small=True,
                                             download=True,
                                             transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))
                dataset = ReturnIndexDataset(dataset, transform=None)
                return dataset

        elif self.dataset_params['dataset_name'] == 'Tiny-ImageNet' or self.dataset_params['dataset_name'] == 'ImageNet':
            # ImageNet
            # Train: 1,281,167 images
            # Val: 50,000 images
            # Test: 100,000 images
            # Tiny-ImageNet: All images are 64x64 colored ones, in 200 classes
            # Train: 500*200 images = 100,000 images
            # Val: 50*200 images = 10,000 images
            # Test: 50*200 images (unlabeled) = 10,000 images

            if self.dataset_params['dataset_name'] == 'Tiny-ImageNet':
                train_folder = os.path.join(self.dataset_params['data_folder'], 'tiny-imagenet-200/train/')
                val_folder = os.path.join(self.dataset_params['data_folder'], 'tiny-imagenet-200/val/')
                valid_size = 0.1
            elif self.dataset_params['dataset_name'] == 'ImageNet':
                train_folder = os.path.join(self.dataset_params['data_folder'], 'imagenet/train_blurred/')
                val_folder = os.path.join(self.dataset_params['data_folder'], 'imagenet/val_blurred/')
                valid_size = 0.01

            if official_split == 'train/' or official_split == 'val/':
                original_train_dataset = datasets.ImageFolder(train_folder,
                                                              transform=None)
                num_train = len(original_train_dataset)
                split = int(np.floor(valid_size * num_train))
                train_set, valid_set = torch.utils.data.random_split(original_train_dataset, [num_train-split, split],
                                                                     generator=torch.Generator().manual_seed(42))

                train_set = ReturnIndexDataset(train_set, transform=torchvision.transforms.Compose([self.transforms_aug, self.normalize]))
                valid_set = ReturnIndexDataset(valid_set, transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))

                if helper.is_main_process():
                    print(f"There are {len(train_set)} samples in train split, on each rank. ")
                    print(f"There are {len(valid_set)} samples in val split, on each rank. ")
                return train_set, valid_set

            else:
                dataset = ImageFolderReturnIndexDataset(val_folder,
                                                        transform=torchvision.transforms.Compose([self.transforms_plain, self.normalize]))
                return dataset


