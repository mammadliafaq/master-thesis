import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import AverageMeter, get_accuracy


def train_epoch(
    train_loader,
    image_model,
    text_model,
    multi_model,
    criterion,
    optimizer,
    device,
    scheduler,
    epoch,
):
    image_model.train()
    text_model.train()
    multi_model.train()

    loss_score = AverageMeter()
    acc_meter = AverageMeter()

    tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
    for bi, batch in tk0:
        batch_size = batch["label"].shape[0]

        image = batch["image"]
        input_ids = batch["text"][0]
        attention_mask = batch["text"][1]
        targets = batch["label"]

        images = image.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        image_embedding = image_model.extract_features(images)
        image_embedding_normalized = F.normalize(image_embedding, dim=1)

        text_embeddings = text_model.extract_features(input_ids, attention_mask)
        text_embedding_normalized = F.normalize(text_embeddings, dim=1)

        output = multi_model(
            image_embedding_normalized, text_embedding_normalized, targets
        )

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)

        accuracy = get_accuracy(output.detach(), targets)
        acc_meter.update(accuracy.item(), batch_size)

        tk0.set_postfix(
            Train_Loss=loss_score.avg,
            Train_Acc=acc_meter.avg,
            Epoch=epoch,
            LR=optimizer.param_groups[0]["lr"],
        )

        if scheduler is not None:
            scheduler.step()

    return loss_score.avg, acc_meter.avg


def eval_epoch(valid_loader, image_model, text_model, multi_model, criterion, device):
    image_model.eval()
    text_model.eval()
    multi_model.eval()

    loss_score = AverageMeter()
    acc_meter = AverageMeter()

    tk0 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    with torch.no_grad():
        for bi, batch in tk0:
            batch_size = batch["label"].shape[0]

            image = batch["image"]
            input_ids = batch["text"][0]
            attention_mask = batch["text"][1]
            targets = batch["label"]

            images = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            image_embedding = image_model.extract_features(images)
            image_embedding_normalized = F.normalize(image_embedding, dim=1)

            text_embeddings = text_model.extract_features(input_ids, attention_mask)
            text_embedding_normalized = F.normalize(text_embeddings, dim=1)

            output = multi_model(
                image_embedding_normalized, text_embedding_normalized, targets
            )

            loss = criterion(output, targets)

            loss_score.update(loss.detach().item(), batch_size)

            accuracy = get_accuracy(output.detach(), targets)
            acc_meter.update(accuracy.item(), batch_size)

            tk0.set_postfix(Eval_Loss=loss_score.avg, Eval_Acc=acc_meter.avg)

    return loss_score.avg, acc_meter.avg
