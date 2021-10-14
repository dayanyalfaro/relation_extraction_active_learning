import torch
from torch import nn
from transformers.file_utils import PT_SAMPLE_DOCSTRINGS
from . import BasicModule
from transformers import AutoConfig, AutoModel
from utils import seq_len_to_mask


class EncoderModel(BasicModule):
    def __init__(self, cfg):
        super(EncoderModel, self).__init__()
        if cfg.model.offline:
            config = AutoConfig.from_pretrained(cfg.cwd + '/embedding/bert-base-multilingual-cased/', num_labels=cfg.corpus.num_relations)
            self.encoder = AutoModel.from_pretrained(cfg.cwd + '/embedding/bert-base-multilingual-cased/', config=config)
        else:
            config = AutoConfig.from_pretrained(cfg.model.lm_file, num_labels=cfg.corpus.num_relations)
            self.encoder = AutoModel.from_pretrained(cfg.model.lm_file, config=config)
        hidden_size = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(hidden_size, cfg.corpus.num_relations)
        )

    def forward(self, x):
        word, lens = x['word'], x['lens']
        ss, os = x['head_start'], x['tail_start']
        mask = seq_len_to_mask(lens, mask_pos_to_true=False)
        outputs = self.encoder(
            word,
            attention_mask=mask,
        )
        pooled_output = outputs.last_hidden_state
        idx = torch.arange(word.size(0)).to(word.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)

        return logits
