import torch
import pickle
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)

def manual_seed(num: int = 1) -> None:
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)

def load_pkl(fp, verbose=True):
    if verbose:
        logger.info(f'load data from {fp}')
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data, fp, verbose: bool = True):
    if verbose:
        logger.info(f'save data in {fp}')

    with open(fp, 'wb') as f:
        pickle.dump(data, f)

def seq_len_to_mask(seq_len, max_len=None, mask_pos_to_true=True):
    """
    Convert a one-dimensional array representing sequence length to a two-dimensional mask, the default position of the pad is 1.
    Convert 1-d seq_len to 2-d mask.

    :param list, np.ndarray, torch.LongTensor seq_len: shape will be (B,)
    :param int max_len: Pad the length to this length. The default (None) uses the longest length in seq_len. But in the scenario of nn.DataParallel, seq_len of different cards may have
        Difference, so you need to pass in a max_len so that the length of the mask is from pad to that length.
    :return: np.ndarray, torch.Tensor 。shape will be (B, max_length)， The element is similar to bool or torch.uint8
    """
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask