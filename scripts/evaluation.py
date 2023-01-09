import os
import ast
import math
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, top_k_accuracy_score

from helper import accuracy
import helper

@torch.no_grad()
def public_validate_network(rank, val_dataloader, model, dataset_params, keep_rate, random_keep_rate=False):
    device = torch.device("cuda:{}".format(rank))
    model.eval()
    metric_logger = helper.MetricLogger(delimiter="  ")
    header = 'Val:'

    dataset_name = dataset_params['dataset_name']

    all_pred_list = []
    all_true_list = []
    all_loss_list = []
    all_ind_list = []
    count = 0

    for images, labels, ind in metric_logger.log_every(val_dataloader, 40, header):
        # Move images and labels to gpu
        images = images.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True).to(torch.int64)
        labels = torch.nn.functional.one_hot(labels, num_classes=dataset_params['dataset_choice'][dataset_name]['num_classes']).float()

        ind = ind.to(device, non_blocking=True).to(torch.int64)

        # Forward
        with torch.no_grad():
            output = model(rank, images, keep_rate, random_keep_rate)
            if isinstance(output, list):
                for index in range(len(output)):
                    output[index] = torch.nn.functional.log_softmax(output[index], dim=1)
                    output[index] = torch.exp(output[index])
                output = torch.mean(torch.stack(output, dim=0), dim=0)
            else:
                output = torch.nn.functional.log_softmax(output, dim=1)
                output = torch.exp(output)

            loss = -torch.mean(torch.sum(torch.log(output)*labels, dim=1))
            labels = torch.argmax(labels, dim=1)

        if isinstance(labels.squeeze().tolist(), list):
            all_true_list.extend(labels.squeeze().tolist())
        else:
            all_true_list.extend([labels.squeeze().tolist()])

        if isinstance(ind.squeeze().tolist(), list):
            all_ind_list.extend(ind.squeeze().tolist())
        else:
            all_ind_list.extend([ind.squeeze().tolist()])

        all_pred_list.extend(output)
        all_loss_list.extend([loss])
        count += 1

    # we have to create enough room to store the collected objects
    all_pred_list_outputs = [None for _ in range(helper.dist.get_world_size())]
    all_true_list_outputs = [None for _ in range(helper.dist.get_world_size())]
    all_loss_list_outputs = [None for _ in range(helper.dist.get_world_size())]
    all_ind_list_outputs = [None for _ in range(helper.dist.get_world_size())]

    # the first argument is the collected lists, the second argument is the data unique in each process
    dist.all_gather_object(all_pred_list_outputs, all_pred_list)
    dist.all_gather_object(all_true_list_outputs, all_true_list)
    dist.all_gather_object(all_loss_list_outputs, all_loss_list)
    dist.all_gather_object(all_ind_list_outputs, all_ind_list)

    all_pred_list_outputs = [item.tolist() for sublist in all_pred_list_outputs for item in sublist]
    all_true_list_outputs = [item for sublist in all_true_list_outputs for item in sublist]
    all_loss_list_outputs = [item.cpu() for sublist in all_loss_list_outputs for item in sublist]
    all_ind_list_outputs = [item for sublist in all_ind_list_outputs for item in sublist]
    loss_to_return = sum(all_loss_list_outputs)/len(all_loss_list_outputs)

    # create a dataframe
    new_df = pd.DataFrame(list(zip(all_ind_list_outputs, all_true_list_outputs, all_pred_list_outputs)), columns=['ind', 'true', 'pred'])

    acc1 = top_k_accuracy_score(all_true_list_outputs, np.array(all_pred_list_outputs), k=1)
    acc5 = top_k_accuracy_score(all_true_list_outputs, np.array(all_pred_list_outputs), k=5)

    if rank == 0:
        print(f'Gathering image level - Number of validation images {len(all_true_list_outputs)}')

    return loss_to_return, acc1, acc5


@torch.no_grad()
def csaw_validate_for_image_network(rank, val_dataloader, model, keep_rate):
    device = torch.device("cuda:{}".format(rank))
    model.eval()
    metric_logger = helper.MetricLogger(delimiter="  ")
    header = 'Val:'

    all_pred_list = []
    all_true_list = []
    all_basenames_list = []
    all_exams_list = []
    all_loss_list = []
    count = 0

    for images, labels, basenames, exam_notes in metric_logger.log_every(val_dataloader, 40, header):
        # Move images and labels to gpu
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).to(torch.int64)

        # Forward
        with torch.no_grad():
            output = model(rank, images, keep_rate, if_train=False)
            defined_loss = torch.nn.CrossEntropyLoss()
            loss = defined_loss(output, labels)

            softmax = nn.Softmax(dim=1)  # softmax outputs
            output = softmax(output)[:, 1]  # list of length b
            output = output.tolist()

        if isinstance(labels.squeeze().tolist(), list):
            all_true_list.extend(labels.squeeze().tolist())
        else:
            all_true_list.extend([labels.squeeze().tolist()])
        all_pred_list.extend(output)
        all_basenames_list.extend(basenames)
        all_exams_list.extend(exam_notes)
        all_loss_list.extend([loss])
        count += 1

    # we have to create enough room to store the collected objects
    all_pred_list_outputs = [None for _ in range(helper.dist.get_world_size())]
    all_true_list_outputs = [None for _ in range(helper.dist.get_world_size())]
    all_basenames_list_outputs = [None for _ in range(helper.dist.get_world_size())]
    all_exams_list_outputs = [None for _ in range(helper.dist.get_world_size())]
    all_loss_list_outputs = [None for _ in range(helper.dist.get_world_size())]

    # the first argument is the collected lists, the second argument is the data unique in each process
    dist.all_gather_object(all_pred_list_outputs, all_pred_list)
    dist.all_gather_object(all_true_list_outputs, all_true_list)
    dist.all_gather_object(all_basenames_list_outputs, all_basenames_list)
    dist.all_gather_object(all_exams_list_outputs, all_exams_list)
    dist.all_gather_object(all_loss_list_outputs, all_loss_list)

    all_pred_list_outputs = [item for sublist in all_pred_list_outputs for item in sublist]
    all_true_list_outputs = [item for sublist in all_true_list_outputs for item in sublist]
    all_basenames_list_outputs = [item for sublist in all_basenames_list_outputs for item in sublist]
    all_exams_list_outputs = [item for sublist in all_exams_list_outputs for item in sublist]
    all_loss_list_outputs = [item.cpu() for sublist in all_loss_list_outputs for item in sublist]
    # print(all_loss_list_outputs, len(all_loss_list_outputs))
    loss_to_return = sum(all_loss_list_outputs)/len(all_loss_list_outputs)

    # create a dataframe
    new_df = pd.DataFrame(list(zip(all_basenames_list_outputs, all_exams_list_outputs, all_true_list_outputs, all_pred_list_outputs)), columns =['basename', 'exam_note', 'true', 'pred'])
    # drop duplicates on basename column
    new_df = new_df.drop_duplicates(subset=['basename'], keep='last').reset_index(drop=True)
    # calculate image level metric
    img_auc = roc_auc_score(new_df['true'].tolist(), new_df['pred'].tolist())
    true_list_img_level = new_df['true'].tolist()
    # calculate exam level aucs
    new_df['pred_list'] = new_df['exam_note'].map(new_df.groupby('exam_note')['pred'].apply(list).to_dict())
    new_df['true_list'] = new_df['exam_note'].map(new_df.groupby('exam_note')['true'].apply(list).to_dict())
    new_df['pred_list_count'] = new_df['exam_note'].map(new_df.groupby('exam_note').size().to_dict())
    new_df = new_df[new_df['pred_list_count'] == 4].reset_index(drop=True)
    new_df['avg'] = new_df.apply(lambda row: sum(row.pred_list)/len(row.pred_list), axis=1)
    new_df['max'] = new_df.apply(lambda row: max(row.pred_list), axis=1)
    new_df['min'] = new_df.apply(lambda row: min(row.pred_list), axis=1)
    new_df['true'] = new_df.apply(lambda row: max(row.true_list), axis=1)
    new_df = new_df.sort_values(by=['exam_note']).reset_index(drop=True)
    new_df = new_df[['exam_note', 'true', 'avg', 'max', 'min']].drop_duplicates(subset=['exam_note'], keep='last').reset_index(drop=True)

    avg_auc = roc_auc_score(new_df['true'].tolist(), new_df['avg'].tolist())
    max_auc = roc_auc_score(new_df['true'].tolist(), new_df['max'].tolist())
    min_auc = roc_auc_score(new_df['true'].tolist(), new_df['min'].tolist())
    true_list_exam_level = new_df['true'].tolist()

    if rank == 0:
        print(f'Gathering image level - Number of validation images {len(true_list_img_level)}')
        print('Gathering image level - Number of pos/neg: ', sum(true_list_img_level), len(true_list_img_level)-sum(true_list_img_level))

        print(f'Gathering exam level - Number of validation exams {len(true_list_exam_level)}')
        print('Gathering exam level - Number of pos/neg: ', sum(true_list_exam_level), len(true_list_exam_level)-sum(true_list_exam_level))
    return loss_to_return, img_auc, avg_auc, max_auc, min_auc



