import ast
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from timm.data import Mixup
from einops.layers.torch import Rearrange
torch.multiprocessing.set_sharing_strategy('file_system')
from torchviz import make_dot
import helper


def train_for_image_one_epoch(rank, epoch, num_epochs,
                              model, defined_loss, data_loader,
                              optimizer, lr_schedule, clip_grad,
                              keep_rate=1, random_keep_rate=False, accum_iter=1):
    device = torch.device("cuda:{}".format(rank))
    metric_logger = helper.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, num_epochs)
    count = 0
    model.train()

    if rank == 0:
        print('Starting one epoch... ')

    for it, data in enumerate(metric_logger.log_every(iterable=data_loader, print_freq=20, header=header)):
        images = data[0]
        labels = data[1]
        # Get the learning rate based on the current iteration number
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        # Move images and labels to gpu
        images = images.to(device, non_blocking=True)
        labels = labels.type(torch.LongTensor)  # <---- Here (casting)
        labels = labels.to(device, non_blocking=True)

        # Model forward passes + compute the loss
        fp16_scaler = None
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            x = model(rank, images, keep_rate, random_keep_rate)  # x: logits right after the fc layer
            loss = defined_loss['classification_loss'](x, labels)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        if accum_iter == 1:
            # Update network's parameters
            # Clear
            optimizer.zero_grad()
            param_norms = None
            # Fill - backward pass
            loss.backward()
            if clip_grad:
                param_norms = helper.clip_gradients(model, clip_grad)
            # Use
            optimizer.step()

            # Logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
            count += 1
        else:
            # Fill - backward pass
            loss.backward()

            # Weights update
            if ((it + 1) % accum_iter == 0) or (it + 1 == len(data_loader)):
                if clip_grad:
                    param_norms = helper.clip_gradients(model, clip_grad)

                # Use
                optimizer.step()

                # Clear
                optimizer.zero_grad()

                # Logging
                torch.cuda.synchronize()
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
                count += 1

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    global_avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return global_avg_stats



