import torch
import heapq
import random
import logging
import numpy as np

from torch import nn
from torch._C import preserve_format
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod
from sklearn.neighbors import NearestNeighbors

from dataset import collate_fn
from models.LMModel import LMFcExtractor

logger = logging.getLogger(__name__)


class QueryBase(metaclass=ABCMeta):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.num_relations = cfg.corpus.num_relations
        self.batch_size = cfg.select_batch_size
        self.pre_batch_size = cfg.pre_batch_size
        self.balance = cfg.balance
        self.class_strategy = cfg.class_strategy
        self.neighbors = cfg.neighbors

    def __call__(self, cur_labeled_ds, unlabeled_ds, model):
        pre_select, values = self.pre_sample(cur_labeled_ds, unlabeled_ds, model)

        if self.balance:
            classes = self.predict_classes(pre_select, cur_labeled_ds, unlabeled_ds, model)
            balance_weights = self.get_balance_weights(cur_labeled_ds)
            pre_select = self.balance_sample(pre_select, values, classes, balance_weights)

        select = pre_select[:self.batch_size]
        return self.get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)

    @abstractmethod
    def pre_sample(self, *args):
        pass

    def predict_classes(self,pre_select, cur_labeled_ds, unlabeled_ds, model):
        classes = {}
        extractor = LMFcExtractor(model)

        if self.class_strategy == 'knn':
            features = {}

            for index, rel in cur_labeled_ds.items():
                features[index] = {}
                one_dataloader = DataLoader([rel], collate_fn=collate_fn(self.cfg))
                (x, y) = next(iter(one_dataloader))
                for key, value in x.items():
                    x[key] = value.to(self.device)
                    with torch.no_grad():
                        feature = extractor(x).cpu().detach()
                        features[index]['feature'] = np.array(feature)
                        features[index]['class'] = rel['rel2idx']

            ratio = { index: [] for index in pre_select}

            for j in range(self.num_relations):
                class_j = []
                class_no_j = []

                for rel in features.values():
                    if rel['class'] == j:
                        class_j.append(rel['feature'])
                    else:
                        class_no_j.append(rel['feature'])

                nbrs_j = NearestNeighbors(n_neighbors=self.neighbors, algorithm='auto').fit(np.array(class_j))
                nbrs_no_j = NearestNeighbors(n_neighbors=self.neighbors, algorithm='auto').fit(np.array(class_no_j))

                for index in pre_select:
                    one_dataloader = DataLoader([unlabeled_ds[index]], collate_fn=collate_fn(self.cfg))
                    (x, y) = next(iter(one_dataloader))
                    for key, value in x.items():
                        x[key] = value.to(self.device)
                        with torch.no_grad():
                            feature = np.array(extractor(x).cpu().detach())

                    d_j, inds_j = nbrs_j.kneighbors(feature)
                    d_no_j, inds_no_j = nbrs_no_j.kneighbors(feature)

                    ratio[index].append(d_j.sum()/d_no_j.sum())

            for index in pre_select:
                lst = ratio[index]
                clss = lst.index(min(lst))
                classes[index] = clss

        return classes


    def get_balance_weights(self, cur_labeled_ds):
        total = len(cur_labeled_ds)
        cants = [0] * self.num_relations

        for rel in cur_labeled_ds.values():
            j = rel['rel2idx']
            cants[j] += 1

        weights = [1/(self.num_relations * c_j / total) for c_j in cants]
        return weights

    def balance_sample(self, pre_select, values, classes, balance_weights):
        results = [values[i] * balance_weights[classes[pre_select[i]]] for i in range(len(pre_select))]
        sorted_results = sorted(zip(results, pre_select))
        _, pre_select = list(zip(*sorted_results))
        return pre_select


    def get_divided_by_select(self, cur_labeled_ds, unlabeled_ds, select):
        logger.info(f'select index:{select}')
        for index in select:
            cur_labeled_ds[index] = unlabeled_ds.pop(index)

        return cur_labeled_ds, unlabeled_ds


class QueryRandom(QueryBase):
    def pre_sample(self, cur_labeled_ds, unlabeled_ds, model):
        # Select size samples without repetition to mark
        return random.sample(unlabeled_ds.keys(), self.pre_batch_size), [1] * self.pre_batch_size

class QueryUncertainty(QueryBase):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
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

    def pre_sample(self,cur_labeled_ds, unlabeled_ds, model):
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

# class QueryUncertainty(QueryBase):
#     def __init__(self, cfg, device):
#         super().__init__(cfg, device)
#         self.type = cfg.strategy.type
#         self.cfg = cfg

#     def get_probs_model(self, model, unlabeled_ds):
#         model.eval()
#         all_y_pred = np.empty((0, self.num_relations))
#         for index, one in enumerate(unlabeled_ds):
#             one_dataloader = DataLoader([one], collate_fn=collate_fn(self.cfg))
#             (x, y) = next(iter(one_dataloader))
#             for key, value in x.items():
#                 x[key] = value.to(self.device)
#             with torch.no_grad():
#                 y_pred = model(x)
#             y_pred = y_pred.cpu().detach().numpy()
#             all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)

#         all_y_pred = torch.from_numpy(all_y_pred)
#         all_y_pred_probability = nn.functional.softmax(all_y_pred, dim=1).numpy()
#         return all_y_pred_probability

#     def pre_sample(self,cur_labeled_ds, unlabeled_ds, model):
#         all_y_pred_probability = self.get_probs_model(model, unlabeled_ds)
#         select = None
#         if self.type == 'least_confident':
#             # The most likely label probability predicted by the current model is the smallest
#             tmp = all_y_pred_probability.max(axis=1)
#             select = heapq.nsmallest(self.batch_size, range(len(tmp)), tmp.take)
#         elif self.type == 'margin_sampling':
#             res = np.empty(0)
#             ttmp = np.vsplit(all_y_pred_probability, all_y_pred_probability.shape[0])
#             for tmp in ttmp:
#                 tmp = np.squeeze(tmp)
#                 # Take the two largest numbers
#                 first_two = tmp[np.argpartition(tmp, -2)[-2:]]
#                 # Collect the difference between the largest two numbers
#                 res = np.concatenate((res, np.array([abs(first_two[0] - first_two[1])])), axis=0)
#             select = heapq.nsmallest(self.batch_size, range(len(res)), res.take)
#         elif self.type == 'entropy_sampling':
#             res = np.empty(0)
#             ttmp = np.vsplit(all_y_pred_probability, all_y_pred_probability.shape[0])
#             for tmp in ttmp:
#                 tmp = np.squeeze(tmp)
#                 # .The dot method seems to be able to automatically transpose and then find the dot product
#                 res = np.concatenate((res, np.array([tmp.dot(np.log2(tmp))])), axis=0)
#             select = heapq.nsmallest(self.batch_size, range(len(res)), res.take)
#         else:
#             assert ('uncertainty concrete choose error')

#         return select