import os
import torch
import hydra
import random
import logging
import torch.nn as nn

import models
import data_select

from hydra import utils
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import preprocess
from training import train, validate
from utils import load_pkl, manual_seed
from dataset import CustomDataset, collate_fn

logger = logging.getLogger(__name__)

@hydra.main(config_path='config/', config_name = 'config')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    cfg.pos_size = 2 * cfg.pos_limit + 2
    logger.info(f'\n{OmegaConf.to_yaml(cfg)}')

    __Model__ = {'cnn' : models.CNNModel,
                 'rnn' : models.LSTMModel,
                 'lm'  : models.LMModel,
                 'encoder' : models.EncoderModel
    }

    __Select__ = {
        'random': data_select.QueryRandom,
        'uncertainty': data_select.QueryUncertainty
    }

    # device
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')

    if cfg.preprocess:
        preprocess(cfg)

    train_data_path = os.path.join(cfg.cwd, cfg.corpus.out_path, cfg.model.model_name, 'train.pkl')
    valid_data_path = os.path.join(cfg.cwd, cfg.corpus.out_path, cfg.model.model_name, 'valid.pkl')
    test_data_path = os.path.join(cfg.cwd, cfg.corpus.out_path, cfg.model.model_name, 'test.pkl')
    vocab_path = os.path.join(cfg.cwd, cfg.corpus.out_path, cfg.model.model_name, 'vocab.pkl')

    if cfg.model.model_name in ('lm', 'encoder'):
        vocab_size = None
    else:
        vocab = load_pkl(vocab_path)
        vocab_size = vocab.count
    cfg.vocab_size = vocab_size

    valid_ds = CustomDataset(valid_data_path)
    test_ds = CustomDataset(test_data_path)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
    test_dataloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))

    all_train_ds = load_pkl(train_data_path)
    random.shuffle(all_train_ds)
    all_train_ds = {index: value for index, value in enumerate(all_train_ds)}
    lst = list(all_train_ds.items())
    if cfg.active_learning:
        cur_labeled_ds = dict(lst[:cfg.start_size])
    else:
        cur_labeled_ds = all_train_ds
    unlabeled_ds = dict(lst[cfg.start_size:])

    per_log_num = 400
    all_size = len(all_train_ds)
    print(all_size)
    writer = SummaryWriter('tensorboard')

    query_strategy = __Select__[cfg.strategy.name](cfg, device)

    logger.info('=' * 10 + ' Start training ' + '=' * 10)
    test_f1_scores, test_losses = [], []
    while len(cur_labeled_ds) <= all_size:
        model = __Model__[cfg.model.model_name](cfg)
        if (not cfg.active_learning) or len(cur_labeled_ds) == cfg.start_size:
            logger.info(f'\n {model}')
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience)
        criterion = nn.CrossEntropyLoss()

        train_dataloader = DataLoader(list(cur_labeled_ds.values()), batch_size=cfg.batch_size, shuffle=True,
                                      collate_fn=collate_fn(cfg))

        train_losses, valid_losses, one_f1_scores = [], [], []
        for epoch in range(1, cfg.epoch + 1):
            # Ensure that the random seed is different every round
            manual_seed(cfg.seed + epoch)
            train_loss = train(epoch, model, train_dataloader, optimizer, criterion, device, writer, cfg)
            # Validation set
            valid_f1, valid_loss = validate(epoch, model, valid_dataloader, criterion, device, cfg)
            # Adjust the learning rate according to valid_loss
            scheduler.step(valid_loss)
            # model_path = model.save(epoch, cfg)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            one_f1_scores.append(valid_f1)

        if (not cfg.active_learning) or (cfg.show_plot and cfg.plot_utils == 'tensorboard' and (len(cur_labeled_ds) - cfg.start_size) % per_log_num == 0):
            logger.info(f'one_f1_scores:{one_f1_scores}')
            for i in range(len(train_losses)):
                writer.add_scalars(f'valid_copy/valid_loss_{len(cur_labeled_ds)}', {
                    'train': train_losses[i],
                    'valid': valid_losses[i]
                }, i)
                writer.add_scalars(f'valid/valid_f1_score_{len(cur_labeled_ds)}', {
                    'valid_f1_score': one_f1_scores[i]
                }, i)

        test_f1, test_loss = validate(-1, model, test_dataloader, criterion, device, cfg)
        test_f1_scores.append(test_f1)
        test_losses.append(test_loss)
        # The average of the last 5 iterations is used as the f1_score performance under the current number of samples
        # f1_scores.append(mean(one_f1_scores[-5:]))
        if len(cur_labeled_ds) == all_size:
            break

        cur_labeled_ds, unlabeled_ds = query_strategy(cur_labeled_ds, unlabeled_ds, model)

    if cfg.show_plot and cfg.plot_utils == 'tensorboard':
        for j in range(len(test_f1_scores)):
            writer.add_scalars('test/test_losses', {
                'test_losses': test_losses[j]
            }, j)
            writer.add_scalars('test/test_f1_scores', {
                'test_f1_scores': test_f1_scores[j],
            }, j)
        writer.close()

    # Test set
    # validate(-1, model, test_dataloader, criterion, device, cfg)

if __name__ == '__main__':
    import time

    cur = time.time()
    main()
    logger.info(f'run time:{time.time() - cur}')
