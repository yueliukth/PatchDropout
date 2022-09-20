import os
import sys
import time
import datetime
import random
import math
import json
import yaml
import cv2
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import socket
from copy import deepcopy
from pathlib import Path
from collections import defaultdict, deque
from operator import itemgetter
from typing import Iterator, List, Optional, Union
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
from torchvision import transforms
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset

def unpatchify(x, patch_size):
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def print_model_checkpoint_matching(model, checkpoint):
    if is_main_process():
        print('Keys in loaded checkpoint that do not exist in the model are: ')
        print(sorted(list(set(checkpoint.keys()) - set(model.state_dict().keys()))))
        print('Keys in the model that do not exist in the loaded checkpoint are: ')
        print(sorted(list(set(model.state_dict().keys()) - set(checkpoint.keys()))))


def calc_kendall_rank_correlation(all_preds, all_labels):
    """Gets the kendall's tau-b rank correlation coefficient.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    Parameters
    ----------
    all_preds: list
        A list of predicted values.
    all_labels: list
        A list of labels.
    Returns
    -------
    correlation: float
        The tau statistic.
    pvalue: float
        The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0.
    """

    tau, p_value = stats.kendalltau(all_preds, all_labels)
    return tau

def calc_class_absolute_error(all_preds, all_labels):
    """Gets the average mean absolute error (AMAE).
    Parameters
    ----------
    all_preds: list
        A list of predicted values.
    all_labels: list
        A list of labels.
    Returns
    -------
    amae: float
        The AMAE.
    """

    label_set = list(set(all_labels))
    all_mae = []
    for label in label_set:
        index_list = [i for i, x in enumerate(all_labels) if x == label]
        pred_list = [all_preds[i] for i in index_list]
        label_list = [all_labels[i] for i in index_list]
        mae = mean_absolute_error(pred_list, label_list)
        all_mae.append(mae)
    return np.average(all_mae)

def make_monotonic(cdf_list):
    monotonic = []
    for i in range(7):
        max_cdf = max(cdf_list[i:])
        monotonic.append(max_cdf)
    return monotonic


def make_probs(lst):
    extended = deepcopy(lst)
    extended.insert(0, 1)  # cdf: 1 in the beginning
    extended.append(0)  # cdf: 0 at last

    probs = []
    for i in range(0, len(extended) - 1):
        probs.append(extended[i] - extended[i + 1])
    return probs

def softmax_score(probs_as_list):
    # score = np.sum([((i + 1) * probs_as_list[i]) for i in range(len(probs_as_list))]) / 8
    score = np.sum([(i * probs_as_list[i]) for i in range(len(probs_as_list))]) / 7
    return score

def multi_hot_score(cdf_list):
    monotonic = make_monotonic(cdf_list)  # we get list of 7 cdf values
    probs = make_probs(monotonic)
    score = softmax_score(probs)  # now that we have prob for each bin, we can use the nice formula
    return probs, score

def make_multi_hot(label, n_labels=8):
    multi_hot = [0] * (n_labels - 1)
    if label > 0:
        for i in range(label):
            multi_hot[i] = 1
    return torch.tensor(multi_hot, dtype=torch.float32)

def make_multi_hot_batch(labels, n_labels=8):
    label_tensors = []
    for ind in range(labels.size()[0]):
        label = labels[ind]-1 # map [1,8] to [0,7]
        label_tensors.append(make_multi_hot(label, n_labels))
    return torch.stack(label_tensors, dim=0)

def scan_thresholded(thresh_row):
    predicted_label = 0
    for ind in range(thresh_row.shape[0]):  # start scanning from left to right
        if thresh_row[ind] == 1:
            predicted_label += 1
        else:  # break the first time we see 0
            break
    return predicted_label

def init_ln_module(model):
    for layer in model.modules():
        if isinstance(layer, nn.LayerNorm):
            layer.reset_parameters()

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def segment_breast_torch(img, threshold=0):
    # convert tensor to numpy array, and rearrange the channel order
    img_for_segment = img.cpu().detach().numpy() # dtype float32 from 0.0 to 1.0
    img_for_segment = 255 * img_for_segment # now scale by 255
    img = img_for_segment.astype('uint8') # convert it to uint
    img = np.moveaxis(img, 0, -1)

    _, img_bin = cv2.threshold(src=img, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
    # find the largest contour
    if len(img_bin.shape) == 3:
        find_contour_image = img_bin[:,:,0].copy()
    elif len(img_bin.shape) < 3:
        find_contour_image = img_bin.copy()

    _, contours, _ = cv2.findContours(find_contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cont_areas = [cv2.contourArea(cont) for cont in contours]
    try:
        idx = np.argmax(cont_areas) # image might be empty
        breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, color=255, thickness=-1)  # -1 is the same as cv2.FILLED
    except Exception:
        breast_mask = np.zeros_like(img_bin)
    if len(breast_mask.shape) == 3:
        return breast_mask[:,:,0]
    else:
        return breast_mask


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def save_log_txt(epoch, model_path, train_global_avg_stats, train_median_stats, \
                 train_avg_stats, train_max_stats, train_value_stats, postfix='_finetuning.txt'):
    train_global_avg_log = {**{f'train_{k}': v for k, v in train_global_avg_stats.items()}, 'epoch': epoch}
    train_median_log = {**{f'train_{k}': v for k, v in train_median_stats.items()}, 'epoch': epoch}
    train_avg_log = {**{f'train_{k}': v for k, v in train_avg_stats.items()}, 'epoch': epoch}
    train_max_log = {**{f'train_{k}': v for k, v in train_max_stats.items()}, 'epoch': epoch}
    train_value_log = {**{f'train_{k}': v for k, v in train_value_stats.items()}, 'epoch': epoch}

    with (Path(model_path + "train_global_avg_log"+postfix)).open("a") as f:
        f.write(json.dumps(train_global_avg_log) + "\n")
    with (Path(model_path + "train_median_log"+postfix)).open("a") as f:
        f.write(json.dumps(train_median_log) + "\n")
    with (Path(model_path + "train_avg_log"+postfix)).open("a") as f:
        f.write(json.dumps(train_avg_log) + "\n")
    with (Path(model_path + "train_max_log"+postfix)).open("a") as f:
        f.write(json.dumps(train_max_log) + "\n")
    with (Path(model_path + "train_value_log"+postfix)).open("a") as f:
        f.write(json.dumps(train_value_log) + "\n")

def log_train_stats(writer, epoch, train_global_avg_stats, train_median_stats, train_avg_stats, train_max_stats, train_value_stats):
    print('Training one epoch is done, start writing loss and learning rate in tensorboard...')
    writer.add_scalar("globalavg_train_loss", train_global_avg_stats['loss'], epoch)
    writer.add_scalar("globalavg_train_lr", train_global_avg_stats['lr'], epoch)
    writer.add_scalar("globalavg_train_wd", train_global_avg_stats['wd'], epoch)
    writer.add_scalar("median_train_loss", train_median_stats['loss'], epoch)
    writer.add_scalar("median_train_lr", train_median_stats['lr'], epoch)
    writer.add_scalar("median_train_wd", train_median_stats['wd'], epoch)
    writer.add_scalar("avg_train_loss", train_avg_stats['loss'], epoch)
    writer.add_scalar("avg_train_lr", train_avg_stats['lr'], epoch)
    writer.add_scalar("avg_train_wd", train_avg_stats['wd'], epoch)
    writer.add_scalar("max_train_loss", train_max_stats['loss'], epoch)
    writer.add_scalar("max_train_lr", train_max_stats['lr'], epoch)
    writer.add_scalar("max_train_wd", train_max_stats['wd'], epoch)
    writer.add_scalar("value_train_loss", train_value_stats['loss'], epoch)
    writer.add_scalar("value_train_lr", train_value_stats['lr'], epoch)
    writer.add_scalar("value_train_wd", train_value_stats['wd'], epoch)

def log_image_tb(writer, epoch, images, name, num_channels, dataset_name):
    if dataset_name == 'ImageNet':
        invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                       transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ]),
                                       ])
    elif dataset_name == 'CSAW' or dataset_name == 'CSAW-M':
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                             std = [ 1/0.16129874, 1/0.16129874, 1/0.16129874 ]),
                                        transforms.Normalize(mean = [ -0.08439291, -0.08439291, -0.08439291 ],
                                                             std = [ 1., 1., 1. ]),
                                        ])

    if num_channels == 3:
        grid = torchvision.utils.make_grid(invTrans(images))
        writer.add_image(name, grid, epoch)
    else:
        writer.add_image(name, images[0,:,:,:], epoch)

def load_state_dict(model, state_dict, skip_key_list=(), skip_key_contains=None, switch_from_ddp=True, strict=True):
    if skip_key_contains!=None or len(skip_key_list)!=0:
        if not skip_key_contains:
            state_dict = {k:v for k, v in state_dict.items() if k not in skip_key_list}
        else:
            state_dict = {k:v for k, v in state_dict.items() if k not in skip_key_list and skip_key_contains not in k}

    if switch_from_ddp:
        # remove `module.` prefix if we need to switch from ddp
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=strict)
    if is_main_process():
        print('Pretrained weights loaded with msg: {}'.format(msg), end='\n\n')

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                if is_main_process():
                    print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    if is_main_process():
                        print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    if is_main_process():
                        print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            if is_main_process():
                print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def initialize_momentum_state(online_net: nn.Module, momentum_net: nn.Module):
    """Copies the parameters of the online network to the momentum network.
    Args:
        online_net (nn.Module): online network (e.g. online encoder, online projection, etc...).
        momentum_net (nn.Module): momentum network (e.g. momentum encoder,
            momentum projection, etc...).
    """
    # online_net and momentum_net start with the same weights
    # Both parameters and persistent buffers (e.g. running averages) are included in state_dict
    momentum_net.load_state_dict(online_net.module.state_dict())
    # There is no backpropagation through the momentum net, so no need for gradients
    # This step is to save some memory
    for param in momentum_net.parameters():
        param.requires_grad = False

def clip_gradients(model, clip):
    """Rescale norm of computed gradients.
    Parameters
    ----------
    model : nn.Module
        Module.
    clip : float
        Maximum norm.
    """
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            print(n)
            p.grad = None

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def constant_scheduler(base_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, step_epoch=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if step_epoch < epochs:
        iters1 = step_epoch * niter_per_ep - warmup_iters
        iters2 = epochs * niter_per_ep - step_epoch * niter_per_ep
        schedule = [base_value]*iters1 + [base_value/10]*iters2
    else:
        iters1 = epochs * niter_per_ep - warmup_iters
        schedule = [base_value]*iters1

    if warmup_epochs > 0:
        schedule = np.concatenate((warmup_schedule, schedule))

    # print(schedule)
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def get_num_layer_for_vit(var_name, num_max_layer):
    if 'cls_token' in var_name \
            or 'pos_embed' in var_name \
            or 'patch_embed' in var_name:
        return 0
    elif var_name.startswith("module.encoder.blocks"):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))

def get_params_groups(model, weight_decay):
    # When using per-parameter options,
    # Each of them will define a separate parameter group,
    # and should contain a params key, containing a list of parameters belonging to it.
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        # if 'cuda' not in str(param.device):
        #     print(name)
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized, 'weight_decay': weight_decay}, {'params': not_regularized, 'weight_decay': 0.}]

def concat_generators(*args):
    for gen in args:
        yield from gen

def get_params_per_model(model, weight_decay, skip_list, get_num_layer, get_layer_scale, if_encoder=True):
    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if if_encoder:
            if get_num_layer is not None:
                layer_id = get_num_layer(name)
                group_name = "layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None
        else:
            group_name = 'classifier_'+group_name

        if group_name not in parameter_group_names:
            if if_encoder:
                if get_layer_scale is not None:
                    scale = get_layer_scale(layer_id)
                else:
                    scale = 1.
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return parameter_group_vars, parameter_group_names


def layer_decay_get_params_groups(model, weight_decay, skip_list=(), get_num_layer=None, get_layer_scale=None):
    encoder_params, encoder_names = get_params_per_model(model['model'], weight_decay, skip_list, get_num_layer, get_layer_scale, if_encoder=True)
    return list({**encoder_params, }.values())


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def get_open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def set_sys_params(random_seed):
    # Set random seeds for reproducibility TODO: to figure out whether it is necessary to have different random seeds
    # on different ranks (DeiT uses different seeds)
    seed = random_seed  #+ get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True  # benchmark mode is good whenever your input sizes for your network do not vary.

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not ddp():
        return 0
    return dist.get_rank()

def ddp():
    world_size = dist.get_world_size()
    if not dist.is_available() or not dist.is_initialized() or world_size < 2:
        return False
    return True

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def run_distributed_workers(rank, main_func, world_size, dist_url, args):
    # Initialize the process group
    dist.init_process_group(backend="NCCL", init_method=dist_url, world_size=world_size, rank=rank)

    # Synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    if ddp():
        dist.barrier()

    torch.cuda.set_device(rank)
    # print('| distributed init (rank {}): {}'.format(
    #     rank, dist_url), flush=True)

    main_func(rank, args)

def launch(main_func, args=()):
    # Set gpu params
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args['system_params']['gpu_ids']

    world_size = args['system_params']['num_gpus']
    port = get_open_port()
    dist_url = f"tcp://127.0.0.1:{port}"
    # os.environ[
    #     "TORCH_DISTRIBUTED_DEBUG"
    # ] = "INFO"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    mp.spawn(
        run_distributed_workers,
        nprocs=world_size,
        args=(main_func, world_size, dist_url, args),
        daemon=False,
    )

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def synchronize():
    if not ddp():
        return
    dist.barrier()

def dist_gather_tensor(tensor, mode='all', dst_rank=0, concatenate=True, cat_dim=0, group=None):
    if not ddp():
        if not concatenate:
            tensor = [tensor]
        return tensor
    world_size = dist.get_world_size()
    rt = tensor.clone()
    tensor_list = [torch.zeros_like(rt) for _ in range(world_size)]
    if mode == 'all':
        dist.all_gather(tensor_list, rt, group=group)
    else:
        if dist.get_backend() == 'nccl':
            group = dist.new_group(backend="gloo")
        else:
            group = dist.group.WORLD
        dist.gather(rt,
                    gather_list=tensor_list if dist.get_rank() == dst_rank else None,
                    dst=dst_rank, group=group)
        if dist.get_rank() != dst_rank:
            tensor_list = [tensor]
    if concatenate:
        tensor_list = torch.cat(tensor_list, dim=cat_dim)

    return tensor_list

def compute_stats(dataloader, dataset_params, model_params, channels=3, input_type='alltokens_original'):
    from tqdm import tqdm
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    # x_tot_seg = np.zeros(channels)
    # x2_tot_seg = np.zeros(channels)
    for images, _, segs, _ in tqdm(dataloader):
        x_tot += images.mean([0,2,3]).cpu().numpy()
        x2_tot += (images**2).mean([0,2,3]).cpu().numpy()
        if input_type == 'onlyfgnd':
            pass
        # if not if_only_overall:
        #     # images: b, channels, global_size, global_size
        #     # segs: b, p1*p2
        #     global_size = dataset_params['dataset_choice']['CSAW']['global_size']
        #     patch_size = model_params['patch_size']
        #     assert global_size % patch_size == 0
        #     print(segs.size())
        #     segs = torch.reshape(segs, (int(global_size/patch_size), int(global_size/patch_size)))
        #     segs = segs.repeat_interleave(16, axis=1).repeat_interleave(16, axis=2)
        #
        #     images = images[:,0,:,:]
        #     print(images.size())
        #     print(segs.size())
        #     breakpoint()
        #     # images: b, global_size, global_size
        #     # segs: b, p1, p2

    channel_avr = x_tot/len(dataloader)
    channel_std = np.sqrt(x2_tot/len(dataloader) - channel_avr**2)
    return channel_avr, channel_std

def print_layers(model):
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
    print(names)

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")


class RandomChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = random.choice(self.transforms)

    def __call__(self, img):
        return self.t(img)