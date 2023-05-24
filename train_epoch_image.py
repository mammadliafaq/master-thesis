import torch
from tqdm import tqdm

from utils.utils import AverageMeter, get_accuracy, warm_up_lr


def train_epoch(
    config, train_loader, model, criterion, optimizer, device, scheduler, epoch
):
    model.train()
    loss_score = AverageMeter()
    acc_meter = AverageMeter()

    NUM_EPOCH_WARM_UP = config.train.n_epochs // 25
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP

    LR = config.train.optimizer.learning_rate

    tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
    for bi, d in tk0:
        if config.train.warmup:
            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                bi + 1 <= NUM_BATCH_WARM_UP
            ):  # adjust LR for each training batch during warm up
                warm_up_lr(bi + 1, NUM_BATCH_WARM_UP, LR, optimizer)

        batch_size = d[0].shape[0]

        images = d[0]
        targets = d[1]

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(images, targets)

        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        accuracy = get_accuracy(output.detach(), targets)
        acc_meter.update(accuracy.item(), batch_size)

        tk0.set_postfix(
            tk0.set_postfix(
                Train_Loss=loss_score.avg,
                Train_Acc=acc_meter.avg,
                Epoch=epoch,
                LR=optimizer.param_groups[0]["lr"],
            )
        )

    if scheduler is not None:
        scheduler.step()

    return loss_score.avg, acc_meter.avg


def eval_epoch(data_loader, model, criterion, device):
    model.eval()
    loss_score = AverageMeter()
    acc_meter = AverageMeter()

    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))

    with torch.no_grad():
        for bi, d in tk0:
            batch_size = d[0].size()[0]

            image = d[0]
            targets = d[1]

            image = image.to(device)
            targets = targets.to(device)

            output = model(image, targets)

            loss = criterion(output, targets)

            loss_score.update(loss.detach().item(), batch_size)
            accuracy = get_accuracy(output.detach(), targets)
            acc_meter.update(accuracy.item(), batch_size)

            tk0.set_postfix(Eval_Loss=loss_score.avg, Eval_Acc=acc_meter.avg)

    return loss_score.avg, acc_meter.avg
