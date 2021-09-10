import torch
import heapq
import random
import logging
import numpy as np

from torch import nn
from torch.utils.data import DataLoader

from dataset import collate_fn

logger = logging.getLogger(__name__)

def get_divided_by_select(cur_labeled_ds, unlabeled_ds, select):
    logger.info(f'select index:{select}')
    new_unlabeled_ds = []
    for index, sen in enumerate(unlabeled_ds):
        if index in select:
            cur_labeled_ds.append(sen)
        else:
            new_unlabeled_ds.append(sen)
    return cur_labeled_ds, new_unlabeled_ds

def random_sample(cur_labeled_ds, unlabeled_ds, model, device, cfg):
    # Select size samples without repetition to mark
    select = random.sample(range(0, len(unlabeled_ds)), cfg.select_batch_size)
    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)

def uncertainty_sample(cur_labeled_ds, unlabeled_ds, model, device, cfg):
    model.eval()
    all_y_pred = np.empty((0, cfg.num_relations))
    for index, one in enumerate(unlabeled_ds):
        one_dataloader = DataLoader([one], collate_fn=collate_fn(cfg))
        (x, y) = next(iter(one_dataloader))
        for key, value in x.items():
            x[key] = value.to(device)
        with torch.no_grad():
            y_pred = model(x)
        y_pred = y_pred.cpu().detach().numpy()
        all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)

    all_y_pred = torch.from_numpy(all_y_pred)
    all_y_pred_probability = nn.functional.softmax(all_y_pred, dim=1).numpy()
    type = cfg.concrete
    select = None
    if type == 'least_confident':
        # The most likely label probability predicted by the current model is the smallest
        tmp = all_y_pred_probability.max(axis=1)
        select = heapq.nsmallest(cfg.select_batch_size, range(len(tmp)), tmp.take)
    elif type == 'margin_sampling':
        res = np.empty(0)
        ttmp = np.vsplit(all_y_pred_probability, all_y_pred_probability.shape[0])
        for tmp in ttmp:
            tmp = np.squeeze(tmp)
            # Take the two largest numbers
            first_two = tmp[np.argpartition(tmp, -2)[-2:]]
            # Collect the difference between the largest two numbers
            res = np.concatenate((res, np.array([abs(first_two[0] - first_two[1])])), axis=0)
        select = heapq.nsmallest(cfg.select_batch_size, range(len(res)), res.take)
    elif type == 'entropy_sampling':
        res = np.empty(0)
        ttmp = np.vsplit(all_y_pred_probability, all_y_pred_probability.shape[0])
        for tmp in ttmp:
            tmp = np.squeeze(tmp)
            # .The dot method seems to be able to automatically transpose and then find the dot product
            res = np.concatenate((res, np.array([tmp.dot(np.log2(tmp))])), axis=0)
        select = heapq.nsmallest(cfg.select_batch_size, range(len(res)), res.take)
    else:
        assert ('uncertainty concrete choose error')

    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)