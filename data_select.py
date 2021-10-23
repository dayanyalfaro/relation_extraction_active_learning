import math
import torch
import heapq
import random
import logging
import numpy as np

from torch import nn
from sklearn.cluster import KMeans
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
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
        pre_select = self.pre_sample(cur_labeled_ds, unlabeled_ds, model)

        if self.balance:
            classes = self.predict_classes(pre_select, cur_labeled_ds, unlabeled_ds, model)
            balance_weights = self.get_balance_weights(cur_labeled_ds)
            pre_select = self.balance_sample(pre_select, classes, balance_weights)

        select = pre_select[:self.batch_size]
        return self.get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)

    @abstractmethod
    def pre_sample(self, *args):
        pass

    def extract_features(self, model, unlabeled_ds):
        model.eval()
        all_y_pred = torch.empty((0, self.num_relations))
        for index, one in enumerate(unlabeled_ds.values()):
            one_dataloader = DataLoader([one], collate_fn=collate_fn(self.cfg))
            (x, y) = next(iter(one_dataloader))
            for key, value in x.items():
                x[key] = value.to(self.device)
            with torch.no_grad():
                y_pred = model(x)
            y_pred = y_pred.cpu().detach()
            all_y_pred = torch.cat((all_y_pred, y_pred), 0)
        return all_y_pred

    def predict_prob (self, model, unlabeled_ds):
        all_y_pred = self.extract_features(model, unlabeled_ds)
        all_y_pred_probability = nn.functional.softmax(all_y_pred, dim=1)
        return all_y_pred_probability

    def predict_prob_dropout (self, model, unlabeled_ds):
        model.train()
        probs = torch.zeros([len(unlabeled_ds),self.num_relations])

        for i in range(self.n_drop):
            for index, one in enumerate(unlabeled_ds.values()):
                one_dataloader = DataLoader([one], collate_fn=collate_fn(self.cfg))
                (x, y) = next(iter(one_dataloader))
                for key, value in x.items():
                    x[key] = value.to(self.device)
                with torch.no_grad():
                    y_pred = model(x)
                    prob = nn.functional.softmax(y_pred, dim=1)
                probs[index] += prob.cpu()[0]
        probs /= self.n_drop
        return probs

    def predict_classes(self,pre_select, cur_labeled_ds, unlabeled_ds, model):
        classes = {}
        # extractor = LMFcExtractor(model)
        model.eval()

        if self.class_strategy == 'knn':
            features = {}

            for index, rel in cur_labeled_ds.items():
                features[index] = {}
                one_dataloader = DataLoader([rel], collate_fn=collate_fn(self.cfg))
                (x, y) = next(iter(one_dataloader))
                for key, value in x.items():
                    x[key] = value.to(self.device)
                    with torch.no_grad():
                        feature = model(x).cpu().detach()
                        features[index]['feature'] = np.array(feature[0])
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

                if len(class_j):
                    cant = min(len(class_j), len(class_no_j), self.neighbors)
                    nbrs_j = NearestNeighbors(n_neighbors=cant, algorithm='auto').fit(np.array(class_j))
                    nbrs_no_j = NearestNeighbors(n_neighbors=cant, algorithm='auto').fit(np.array(class_no_j))

                    for index in pre_select:
                        one_dataloader = DataLoader([unlabeled_ds[index]], collate_fn=collate_fn(self.cfg))
                        (x, y) = next(iter(one_dataloader))
                        for key, value in x.items():
                            x[key] = value.to(self.device)
                            with torch.no_grad():
                                feature = np.array(model(x).cpu().detach())

                        d_j, inds_j = nbrs_j.kneighbors(feature)
                        d_no_j, inds_no_j = nbrs_no_j.kneighbors(feature)

                        ratio[index].append(d_j.sum()/d_no_j.sum())
                else:
                    for index in pre_select:
                        ratio[index].append(math.inf)

            for index in pre_select:
                lst = ratio[index]
                clss = lst.index(min(lst))
                classes[index] = clss

        elif self.class_strategy == 'prob':
            pre_dict = { idx : unlabeled_ds[idx] for idx in pre_select}
            probs = self.predict_prob(model,pre_dict)

            for idx, item in enumerate(pre_select):
                classes[item] = int(probs[idx].argmax())

        return classes


    def get_balance_weights(self, cur_labeled_ds):
        total = len(cur_labeled_ds)
        cants = [0] * self.num_relations

        for rel in cur_labeled_ds.values():
            j = rel['rel2idx']
            cants[j] += 1

        weights = []

        for c_j in cants:
            if c_j:
                weights.append(1/(self.num_relations * c_j / total))
            else:
                weights.append(math.inf)

        return weights

    def balance_sample(self, pre_select, classes, balance_weights):
        results = [-balance_weights[classes[pre_select[i]]] for i in range(len(pre_select))]
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
        self.n_drop = cfg.strategy.n_drop
        self.use_dropout = cfg.strategy.use_dropout

    def pre_sample(self,cur_labeled_ds, unlabeled_ds, model):
        if self.use_dropout:
            probs = self.predict_prob_dropout(model, unlabeled_ds)
        else:
            probs = self.predict_prob(model, unlabeled_ds)
        select = None
        # values = None
        U = None
        idxs_unlabeled = np.array(list(unlabeled_ds.keys()))

        if self.type == 'least_confident':
            U = probs.max(1)[0]
        elif self.type == 'margin_sampling':
            probs_sorted, idxs = probs.sort(descending=True)
            U = probs_sorted[:, 0] - probs_sorted[:,1]
        elif self.type == 'entropy_sampling':
            log_probs = torch.log(probs)
            U = (probs*log_probs).sum(1)
        else:
            assert ('uncertainty concrete choose error')

        sorted, idxs = U.sort()
        select = idxs_unlabeled[idxs[:self.pre_batch_size]]
        # values = sorted[:self.pre_batch_size]
        return select

class QueryBALD(QueryBase):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.n_drop = cfg.strategy.n_drop
        self.use_dropout = cfg.strategy.use_dropout

    def pre_sample(self,cur_labeled_ds, unlabeled_ds, model):
        if self.use_dropout:
            probs = self.predict_prob_dropout(model, unlabeled_ds)
        else:
            probs = self.predict_prob(model, unlabeled_ds)

        idxs_unlabeled = np.array(list(unlabeled_ds.keys()))

        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        U = entropy2 - entropy1

        sorted, idxs = U.sort()
        select = idxs_unlabeled[idxs[:self.pre_batch_size]]
        # values = sorted[:self.pre_batch_size]
        return select

class QueryKMeans(QueryBase):
    def pre_sample(self,cur_labeled_ds, unlabeled_ds, model):
        idxs_unlabeled = np.array(list(unlabeled_ds.keys()))

        features = self.extract_features(model,unlabeled_ds)
        features = features.numpy()
        cluster_learner = KMeans(n_clusters=self.pre_batch_size)
        cluster_learner.fit(features)

        cluster_idxs = cluster_learner.predict(features)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (features - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(features.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(self.pre_batch_size)])

        return idxs_unlabeled[q_idxs]