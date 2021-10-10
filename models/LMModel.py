from torch import nn
from transformers.file_utils import PT_SAMPLE_DOCSTRINGS
from . import BasicModule
from modules import RNN
from transformers import BertModel
from utils import seq_len_to_mask


class LMModel(BasicModule):
    def __init__(self, cfg):
        super(LMModel, self).__init__()
        if cfg.model.offline:
            self.bert = BertModel.from_pretrained(cfg.cwd + '/embedding/bert-base-multilingual-cased/', num_hidden_layers=cfg.model.num_hidden_layers)
        else:
            self.bert = BertModel.from_pretrained(cfg.model.lm_file, num_hidden_layers=cfg.model.num_hidden_layers)
        self.bilstm = RNN(cfg)
        self.fc = nn.Linear(cfg.model.hidden_size, cfg.corpus.num_relations)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(self, x):
        word, lens = x['word'], x['lens']
        mask = seq_len_to_mask(lens, mask_pos_to_true=False)
        output = self.bert(word, attention_mask=mask)
        out, out_pool = self.bilstm(output.last_hidden_state, lens)
        out_pool = self.dropout(out_pool)
        output = self.fc(out_pool)

        return output

class LMFcExtractor(nn.Module):
    def __init__(self,submodule):
        super(LMFcExtractor, self).__init__()
        self.submodule = submodule

    def forward(self,x):
        word, lens = x['word'], x['lens']
        mask = seq_len_to_mask(lens, mask_pos_to_true=False)
        output = self.submodule.bert(word, attention_mask=mask)
        out, out_pool = self.submodule.bilstm(output.last_hidden_state, lens)
        return out_pool