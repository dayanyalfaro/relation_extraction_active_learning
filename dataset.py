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
        head_pos, tail_pos = [], []
        pcnn_mask = []
        for data in batch:
            # Perform zero-padded operations for sentences of non-maximum length in the current batch
            # TODO Is this batch all data or
            word.append(_padding(data['token2idx'], max_len))
            word_len.append(data['seq_len'])

            y.append(int(data['rel2idx']))

            if cfg.model.model_name not in ('lm', 'encoder'):
                # print(data.items())
                # if 'head_pos' not in d:

                head_pos.append(_padding(data['head_pos'], max_len))
                tail_pos.append(_padding(data['tail_pos'], max_len))

        # The data of each sample is divided by jieba into the ID corresponding to the phrase and the length of the sentence
        #          The label is the relationship ID, such as the ancestral home corresponding to 3
        x['word'] = torch.tensor(word)
        x['lens'] = torch.tensor(word_len)
        y = torch.tensor(y)

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
