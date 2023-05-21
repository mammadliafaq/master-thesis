import argparse
import os
import shutil
import sys

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter

from dataset import ShopeeImageDataset
from models.image_model import ShopeeImageModel, ShopeeImageTransformerModel
from train import eval_epoch, train_epoch
from transforms import get_train_transforms, get_valid_transforms
from utils.utils import (add_weight_decay, convert_dict_to_tuple,
                         get_optimizer, get_scheduler, seed_worker, set_seed)


def run(args: argparse.Namespace) -> None:
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    outdir = os.path.join(config.outdir, config.exp_name)
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    shutil.copy2(args.cfg, outdir)

    tb = SummaryWriter(outdir)

    # Set seed
    set_seed(config.seed)

    # Defining Device
    device_id = config.gpu_id
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("Selected device: ", device)

    data = pd.read_csv(config.dataset.path_to_folds)
    data["filepath"] = data["image"].apply(
        lambda x: os.path.join(config.dataset.root, "train_images", x)
    )

    encoder = LabelEncoder()
    data["label_group"] = encoder.fit_transform(data["label_group"])

    train = data[data["fold"] != 0].reset_index(drop=True)
    valid = data[data["fold"] == 0].reset_index(drop=True)

    print(f"Data size: train shape = {train.shape[0]}, val shape = {valid.shape[0]}")

    # Defining DataSet
    train_dataset = ShopeeImageDataset(
        csv=train,
        transforms=get_train_transforms(config),
    )

    valid_dataset = ShopeeImageDataset(
        csv=valid,
        transforms=get_valid_transforms(config),
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=config.dataset.num_workers,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.dataset.num_workers,
    )

    model_params = {
        "n_classes": config.dataset.num_of_classes,
        "model_name": config.model.model_name,
        "pretrained": config.model.pretrained,
        "use_fc": config.model.use_fc,
        "fc_dim": config.model.fc_dim,
        "dropout": config.model.dropout,
        "loss_module": config.model.loss_module,
        "s": config.head.s,
        "margin": config.head.margin,
        "ls_eps": config.head.ls_eps,
        "theta_zero": config.head.theta_zero,
    }

    # Defining Model for specific fold
    model = ShopeeImageTransformerModel(**model_params, device=device)
    model.to(device)

    # Defining criterion
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    no_decay_parameters, decay_parameters = add_weight_decay(model)
    optimizer = get_optimizer(
        config,
        decay_parameters=decay_parameters,
        no_decay_parameters=no_decay_parameters,
    )

    scheduler = get_scheduler(config, optimizer)

    # THE TRAIN LOOP
    best_loss = 10000
    for epoch in range(config.train.n_epochs):
        train_loss = train_epoch(
            config,
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            scheduler=None,
            epoch=epoch,
        )

        tb.add_scalar("Train Loss", train_loss, epoch + 1)

        valid_loss = eval_epoch(valid_loader, model, criterion, device)

        tb.add_scalar("Val Loss", valid_loss, epoch + 1)

        if valid_loss <= best_loss:
            best_loss = valid_loss
            name_to_save = f"model_{config.model.model_name}_{config.model.loss_module}_val-loss-{valid_loss:.2f}_epoch-{epoch}.pth"
            path_to_weights = os.path.join(outdir, name_to_save)
            torch.save(
                model.state_dict(),
                path_to_weights,
            )
            print(f"Best model found for epoch {epoch}  has been saved!")

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {cur_lr}")
        tb.add_scalar("Learning rate", cur_lr, epoch + 1)

        scheduler.step()

        print(
            f"Epoch {epoch} finished: Train loss = {train_loss:.2f}; Val loss = {valid_loss:.2f}"
        )
    tb.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="Path to config file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments(sys.argv[1:]))
