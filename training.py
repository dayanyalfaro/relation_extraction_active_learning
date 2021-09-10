import torch
import logging
from metric import PRMetric

logger = logging.getLogger(__name__)


def train(epoch, model, dataloader, optimizer, criterion, device, writer, cfg):
    model.train()
    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            # Let the key point to the value after the device is placed
            x[key] = value.to(device)
        y = y.to(device)
        # Clear the gradient of the last iteration
        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss.backward()
        # Update model parameters
        optimizer.step()

        metric.update(y_true=y, y_pred=y_pred)
        losses.append(loss.item())
        # TODO batch_size * 10 correct?
        data_total = len(dataloader.dataset)
        data_cal = data_total if batch_idx == len(dataloader) else batch_idx * len(y)
        if (cfg.train_log and batch_idx % cfg.log_interval == 0) or batch_idx == len(dataloader):
            # p r f1 are all macros, because the three are the same for micro, they are defined as acc
            acc, p, r, f1 = metric.compute()
            logger.info(f'Train Epoch {epoch}: [{data_cal}/{data_total}]({100. * data_cal / data_total:.0f}%)\t'
                        f'Loss: {loss.item():.6f}\t metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    # if cfg.show_plot and not cfg.only_comparison_plot and cfg.plot_utils == 'tensorboard':
    #     for i in range(len(losses)):
    #         writer.add_scalar(f'epoch_{epoch}_training_loss', losses[i], i)

    return sum(losses) / len(losses)


def validate(epoch, model, dataloader, criterion, device, cfg):
    model.eval()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x)

            loss = criterion(y_pred, y)

            metric.update(y_true=y, y_pred=y_pred)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)
    acc, p, r, f1 = metric.compute()
    data_total = len(dataloader.dataset)

    if epoch >= 0:
        logger.info(f'Valid Epoch {epoch}: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}\t'
                    f'metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')
    else:
        logger.info(f'Test Data: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}\t'
                    f'metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    return f1, loss
