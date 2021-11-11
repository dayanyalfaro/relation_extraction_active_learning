import os
import json
import time
import wandb
import torch
import hydra
import random
import logging
import requests
import torch.nn as nn

import models
import data_select

from hydra import utils
from omegaconf import OmegaConf
from alipy.data_manipulate import split, split_load
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import preprocess
from training import train, validate
from utils import load_pkl, manual_seed
from dataset import CustomDataset, collate_fn
from metric import IRMetric, IDMetric, LRIDMetric

logger = logging.getLogger(__name__)

os.environ["WANDB_API_KEY"] = '42b01ac48aabb9117171872873dccfa9e26aa8c0'
os.environ["WANDB_MODE"] = "dryrun"

url = 'https://api.telegram.org/bot2144234362:AAEgU3ZjsZxbrZSYVCUNdor-T-Vxo10wjJY/sendMessage?chat_id=765144530&text='

@hydra.main(config_path='config/', config_name = 'config')
def main(cfg):
    start_time = time.time()
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
        'uncertainty': data_select.QueryUncertainty,
        'bald': data_select.QueryBALD,
        'kmeans': data_select.QueryKMeans
    }

    wandb_config = {
                        'corpus' : cfg.corpus.name,
                        'model' : cfg.model.model_name,
                        'strategy' : cfg.strategy.name + cfg.strategy.type,
                        'balance' : cfg.balance,
                        'balance_strategy' : cfg.class_strategy
    }

    # device
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')

    if cfg.preprocess:
        preprocess(cfg)
        requests.get(url + 'preprocess done')

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
    class_dist = [item['rel2idx'] for item in all_train_ds]
    split_path = cfg.cwd + '/' + cfg.corpus.out_path + '/' + cfg.model.model_name
    if cfg.split:
        logger.info('Splitting dataset into labeled and unlabeled')
        _, _, lab, _ = split( y=class_dist, test_ratio=0, initial_label_rate=0.1,
                                    split_count= cfg.seeds_count, all_class=True, saving_path=split_path)
        requests.get(url + 'split done')
    else:
        logger.info('Loading dataset splits')
        _, _, lab, _ = split_load(split_path)
        if cfg.seeds_count == 1:
            lab = [lab]

    for idx,label_split in enumerate(lab,1):
        summary = {
                    'model' : cfg.model.model_name,
                    'strategy': cfg.strategy.name
                    }

        run_name = f'{cfg.strategy.name}{cfg.strategy.type}_{cfg.select_batch_size}'
        if cfg.balance:
            run_name += '_balance_{cfg.pre_batch_size}'
        run_name += f'_{idx}'

        with open(f'{run_name}.json', 'x') as f:
            json.dump(summary,f)

        run = wandb.init(project="relation_extraction_active_learning", name= run_name, config = wandb_config)
        requests.get(url + f'run {run_name} started')

        if cfg.active_learning:
            cur_labeled_ds = {}
            unlabeled_ds = {}
            for index, value in enumerate(all_train_ds):
                if index in label_split:
                    cur_labeled_ds[index] = value
                else:
                    unlabeled_ds[index] = value
        else:
            cur_labeled_ds = {index: value for index, value in enumerate(all_train_ds)}

        per_log_num = 400
        all_size = len(all_train_ds)
        # print(all_size)
        writer = SummaryWriter('tensorboard')

        query_strategy = __Select__[cfg.strategy.name](cfg, device)

        logger.info('=' * 10 + ' Start training ' + '=' * 10)
        test_f1_scores, test_losses = [], []
        n_iter = 0

        # Balance metrics
        labeled_classes = [value['rel2idx'] for value in cur_labeled_ds.values()]
        IR = IRMetric(labeled_classes)
        ID = IDMetric(labeled_classes)
        LRID = LRIDMetric(labeled_classes)

        while n_iter < cfg.total_iter:
        # while len(cur_labeled_ds) <= all_size:
            n_iter += 1
            summary[n_iter] = {}

            manual_seed(cfg.seed)
            model = __Model__[cfg.model.model_name](cfg)
            if (not cfg.active_learning) or len(cur_labeled_ds) == cfg.start_size:
                logger.info(f'\n {model}')
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience)
            criterion = nn.CrossEntropyLoss()

            train_dataloader = DataLoader(list(cur_labeled_ds.values()), batch_size=cfg.batch_size, shuffle=True,
                                        collate_fn=collate_fn(cfg))

            # Performance metrics
            train_acc, train_p, train_r, train_f1, train_loss = [], [], [], [], []
            valid_acc, valid_p, valid_r, valid_f1, valid_loss = [], [], [], [], []

            for epoch in range(1, cfg.epoch + 1):
                # Ensure that the random seed is different every round
                manual_seed(cfg.seed + epoch)
                t_acc, t_p, t_r, t_f1, t_loss = train(epoch, model, train_dataloader, optimizer, criterion, device, writer, cfg)
                # Validation set
                v_acc, v_p, v_r, v_f1, v_loss = validate(epoch, model, valid_dataloader, criterion, device, cfg)
                # Adjust the learning rate according to valid_loss
                scheduler.step(v_loss)
                # model_path = model.save(epoch, cfg)

                train_acc.append(t_acc)
                train_p.append(t_p)
                train_r.append(t_r)
                train_f1.append(t_f1)
                train_loss.append(t_loss)

                valid_acc.append(v_acc)
                valid_p.append(v_p)
                valid_r.append(v_r)
                valid_f1.append(v_f1)
                valid_loss.append(v_loss)

            # if (not cfg.active_learning) or (cfg.show_plot and cfg.plot_utils == 'tensorboard' and (len(cur_labeled_ds) - cfg.start_size) % per_log_num == 0):
            #     # logger.info(f'one_f1_scores:{valid_f1}')
            #     for i in range(len(train_loss)):
            #         writer.add_scalars(f'valid_copy/valid_loss_{len(cur_labeled_ds)}', {
            #             'train': train_loss[i],
            #             'valid': valid_loss[i]
            #         }, i)
            #         writer.add_scalars(f'valid/valid_f1_score_{len(cur_labeled_ds)}', {
            #             'valid_f1_score': valid_f1[i]
            #         }, i)

            # Train logs
            summary[n_iter]['train'] = {
                'acc' : train_acc,
                'p' : train_p,
                'r' : train_r,
                'f1' : train_f1,
                'loss' : train_loss
            }

            # Valid logs
            summary[n_iter]['valid'] = {
                'acc' : valid_acc,
                'p' :   valid_p,
                'r' :   valid_r,
                'f1' :  valid_f1,
                'loss': valid_loss
            }

            test_acc, test_p, test_r, test_f1, test_loss = validate(-1, model, test_dataloader, criterion, device, cfg)
            test_f1_scores.append(test_f1)
            test_losses.append(test_loss)


            # Test logs
            summary[n_iter]['f1'] = test_f1
            summary[n_iter]['p'] = test_p
            summary[n_iter]['r'] = test_r
            summary[n_iter]['acc'] = test_acc
            summary[n_iter]['loss'] = test_loss

            # The average of the last 5 iterations is used as the f1_score performance under the current number of samples
            # f1_scores.append(mean(one_f1_scores[-5:]))

            if len(cur_labeled_ds) == all_size:
                with open(f'{run_name}.json', 'w') as f:
                    json.dump(summary,f)
                break

            t = time.time()
            cur_labeled_ds, unlabeled_ds, selected_idxs, pred_correct = query_strategy(cur_labeled_ds, unlabeled_ds, model)

            summary[n_iter]['time'] = select_time = time.time() - t
            summary[n_iter]['select'] = selected_idxs

            summary[n_iter]['pred_correct'] = pred_correct

            new_labeled_classes = [cur_labeled_ds[index]['rel2idx'] for index in selected_idxs]
            IR.update(new_labeled_classes)
            ID.update(new_labeled_classes)
            LRID.update(new_labeled_classes)
            summary[n_iter]['IR'] = IR_value = IR.compute()
            summary[n_iter]['ID_HE'] = ID_HE_value = ID.compute('HE')
            summary[n_iter]['ID_TV'] = ID_TV_value = ID.compute('TV')
            summary[n_iter]['LRID'] = LRID_value = LRID.compute()

            with open(f'{run_name}.json', 'w') as f:
                json.dump(summary,f)

            run.log({
                'accuracy' : test_acc,
                'precision' : test_p,
                'recall' : test_r,
                'f1' : test_f1,
                'loss' : test_loss,
                'select_time' : select_time,
                'select': selected_idxs,
                'pred_correct': pred_correct,
                'IR' : IR_value,
                'ID_HE' : ID_HE_value,
                'ID_TV' : ID_TV_value,
                'LRID' : LRID_value,
            },
            step = n_iter
            )

            requests.get(url + f'{n_iter} f1: {test_f1}  lrid: {LRID_value}  pred: {pred_correct}')

        # if cfg.show_plot and cfg.plot_utils == 'tensorboard':
        #     for j in range(len(test_f1_scores)):
        #         writer.add_scalars('test/test_losses', {
        #             'test_losses': test_losses[j]
        #         }, j)
        #         writer.add_scalars('test/test_f1_scores', {
        #             'test_f1_scores': test_f1_scores[j],
        #         }, j)
        #     writer.close()

        summary['total_time'] = time.time() - start_time
        with open(f'{run_name}.json', 'w') as f:
            json.dump(summary,f)
        requests.get(url + f'run {run_name} finished')
        # Test set
        # validate(-1, model, test_dataloader, criterion, device, cfg)

if __name__ == '__main__':

    cur = time.time()
    main()
    logger.info(f'run time:{time.time() - cur}')
