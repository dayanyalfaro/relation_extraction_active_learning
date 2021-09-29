import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BasicModule
from modules import Embedding, CNN
from utils import seq_len_to_mask


class CNNModel(BasicModule):
    def __init__(self, cfg):
        super(CNNModel, self).__init__()
        if cfg.dim_strategy == 'cat':
            cfg.model.in_channels = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.model.in_channels = cfg.word_dim

        self.embedding = Embedding(cfg)
        self.cnn = CNN(cfg)
        self.fc1 = nn.Linear(len(cfg.model.kernel_sizes) * cfg.model.out_channels, cfg.model.intermediate)
        self.fc2 = nn.Linear(cfg.model.intermediate, cfg.corpus.num_relations)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        # mask = seq_len_to_mask(lens)

        inputs = self.embedding(word, head_pos, tail_pos)
        out_pool = self.cnn(inputs)

        output = self.fc1(out_pool)
        output = F.leaky_relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output


class CNNFcExtractor(nn.Module):
    def __init__(self, submodule):
        super(CNNFcExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        mask = seq_len_to_mask(lens)

        x = self.submodule.embedding(word, head_pos, tail_pos)
        x = self.submodule.cnn(x, mask=mask)
        x = self.submodule.fc1(x)
        x = F.leaky_relu(x)
        return x

