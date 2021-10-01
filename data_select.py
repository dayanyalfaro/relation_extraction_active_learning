import torch
import heapq
import random
import logging
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod

from dataset import collate_fn

logger = logging.getLogger(__name__)


class QueryBase(metaclass=ABCMeta):
    def __init__(self, cfg, device):
        self.device = device
        self.batch_size = cfg.select_batch_size

    def __call__(self, cur_labeled_ds, unlabeled_ds, model):
        select = self.sample(cur_labeled_ds, unlabeled_ds, model)
        return self.get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)

    @abstractmethod
    def sample(self, *args):
        pass

    def get_divided_by_select(self, cur_labeled_ds, unlabeled_ds, select):
        logger.info(f'select index:{select}')
        new_unlabeled_ds = []
        for index, sen in enumerate(unlabeled_ds):
            if index in select:
                cur_labeled_ds.append(sen)
            else:
                new_unlabeled_ds.append(sen)
        return cur_labeled_ds, new_unlabeled_ds


class QueryRandom(QueryBase):
    def sample(self, cur_labeled_ds, unlabeled_ds, model):
        # Select size samples without repetition to mark
        return random.sample(range(0, len(unlabeled_ds)), self.batch_size)

class QueryUncertainty(QueryBase):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.num_relations = cfg.corpus.num_relations
        self.type = cfg.strategy.type
        self.cfg = cfg

    def get_probs_model(self, model, unlabeled_ds):
        model.eval()
        all_y_pred = np.empty((0, self.num_relations))
        for index, one in enumerate(unlabeled_ds):
            one_dataloader = DataLoader([one], collate_fn=collate_fn(self.cfg))
            (x, y) = next(iter(one_dataloader))
            for key, value in x.items():
                x[key] = value.to(self.device)
            with torch.no_grad():
                y_pred = model(x)
            y_pred = y_pred.cpu().detach().numpy()
            all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)

        all_y_pred = torch.from_numpy(all_y_pred)
        all_y_pred_probability = nn.functional.softmax(all_y_pred, dim=1).numpy()
        return all_y_pred_probability

    def sample(self,cur_labeled_ds, unlabeled_ds, model):
        all_y_pred_probability = self.get_probs_model(model, unlabeled_ds)
        select = None
        if self.type == 'least_confident':
            # The most likely label probability predicted by the current model is the smallest
            tmp = all_y_pred_probability.max(axis=1)
            select = heapq.nsmallest(self.batch_size, range(len(tmp)), tmp.take)
        elif self.type == 'margin_sampling':
            res = np.empty(0)
            ttmp = np.vsplit(all_y_pred_probability, all_y_pred_probability.shape[0])
            for tmp in ttmp:
                tmp = np.squeeze(tmp)
                # Take the two largest numbers
                first_two = tmp[np.argpartition(tmp, -2)[-2:]]
                # Collect the difference between the largest two numbers
                res = np.concatenate((res, np.array([abs(first_two[0] - first_two[1])])), axis=0)
            select = heapq.nsmallest(self.batch_size, range(len(res)), res.take)
        elif self.type == 'entropy_sampling':
            res = np.empty(0)
            ttmp = np.vsplit(all_y_pred_probability, all_y_pred_probability.shape[0])
            for tmp in ttmp:
                tmp = np.squeeze(tmp)
                # .The dot method seems to be able to automatically transpose and then find the dot product
                res = np.concatenate((res, np.array([tmp.dot(np.log2(tmp))])), axis=0)
            select = heapq.nsmallest(self.batch_size, range(len(res)), res.take)
        else:
            assert ('uncertainty concrete choose error')

        return select