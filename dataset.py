import torch
from torch.utils.data import Dataset
from utils import load_pkl


def collate_fn(cfg):
    def collate_fn_intra(batch):
        batch.sort(key=lambda data: data['seq_len'], reverse=True)

        max_len = batch[0]['seq_len']

        def _padding(x, max_len):
            return x + [0] * (max_len - len(x))

        x, y = dict(), []
        word, word_len = [], []
        head_start, tail_start = [], []  
        head_pos, tail_pos = [], []
        for data in batch:
            # Perform zero-padded operations for sentences of non-maximum length in the current batch
            # TODO Is this batch all data or
            word.append(_padding(data['token2idx'], max_len))
            word_len.append(data['seq_len'])

            if cfg.model.model_name == 'encoder':
                head_start.append(int(data['head_start']))
                tail_start.append(int(data['tail_start']))

            y.append(int(data['rel2idx']))

            if cfg.model.model_name not in ('lm', 'encoder'):
                # print(data.items())
                # if 'head_pos' not in d:

                head_pos.append(_padding(data['head_pos'], max_len))
                tail_pos.append(_padding(data['tail_pos'], max_len))

        x['word'] = torch.tensor(word)
        x['lens'] = torch.tensor(word_len)
        y = torch.tensor(y)

        if cfg.model.model_name == 'encoder':
            x['head_start'] = torch.tensor(head_start)
            x['tail_start'] = torch.tensor(tail_start)

        if cfg.model.model_name not in ('lm', 'encoder'):
            x['head_pos'] = torch.tensor(head_pos)
            x['tail_pos'] = torch.tensor(tail_pos)

        return x, y

    return collate_fn_intra


class CustomDataset(Dataset):
    """Use List to store data by default"""
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)
