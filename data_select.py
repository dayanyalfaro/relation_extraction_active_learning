import random
import logging

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