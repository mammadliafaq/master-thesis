import torch
from tqdm import tqdm

from utils.utils import AverageMeter, get_accuracy


def train_epoch(
    train_loader, model, criterion, optimizer, device, scheduler, epoch
):
    model.train()
    loss_score = AverageMeter()
    acc_meter = AverageMeter()

    tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
    for bi, d in tk0:

        batch_size = d[0].shape[0]

        input_ids = d[0]
        attention_mask = d[1]
        targets = d[2]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(input_ids, attention_mask, targets)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        accuracy = get_accuracy(output.detach(), targets)
        acc_meter.update(accuracy.item(), batch_size)

        tk0.set_postfix(Train_Loss=loss_score.avg, Train_Acc=acc_meter.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

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
            batch_size = d[0].shape[0]

            input_ids = d[0]
            attention_mask = d[1]
            targets = d[2]

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            output = model(input_ids, attention_mask, targets)

            loss = criterion(output, targets)

            loss_score.update(loss.detach().item(), batch_size)
            accuracy = get_accuracy(output.detach(), targets)
            acc_meter.update(accuracy.item(), batch_size)

            tk0.set_postfix(Eval_Loss=loss_score.avg, Eval_Acc=acc_meter.avg)

    return loss_score.avg, acc_meter.avg
