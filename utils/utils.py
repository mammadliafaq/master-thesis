import os
import random
from collections import namedtuple

import numpy as np
import torch


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(config, decay_parameters, no_decay_parameters):
    if config.train.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": decay_parameters,
                    "weight_decay": config.train.optimizer.weight_decay,
                },
                {"params": no_decay_parameters},
            ],
            lr=config.train.optimizer.learning_rate,
            momentum=config.train.optimizer.momentum,
        )
    elif config.train.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": decay_parameters,
                    "weight_decay": config.train.optimizer.weight_decay,
                },
                {"params": no_decay_parameters},
            ],
            lr=config.train.optimizer.learning_rate,
        )
    elif config.train.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay_parameters,
                    "weight_decay": config.train.optimizer.weight_decay,
                },
                {"params": no_decay_parameters},
            ],
            lr=config.train.optimizer.learning_rate,
        )
    else:
        raise Exception(
            "Unknown type of optimizer: {}".format(config.train.optimizer.name)
        )
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            eta_min=config.train.lr_scheduler.eta_min,
            T_max=config.train.lr_scheduler.T_max,
        )
    elif config.train.lr_scheduler.name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.lr_scheduler.step_size,
            gamma=config.train.lr_scheduler.gamma,
        )
    elif config.train.lr_scheduler.name == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.train.lr_scheduler.milestones,
            gamma=config.train.lr_scheduler.gamma,
        )
    else:
        raise Exception(
            "Unknown type of lr schedule: {}".format(config.train.lr_schedule)
        )
    return scheduler


# Do not decay parameters for biases and batch-norm layers
def add_weight_decay(model, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return no_decay, decay


# Adjust LR for each training batch during warm up
def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params["lr"] = batch * init_lr / num_batch_warm_up


def get_accuracy(preds, targets):
    """
    preds shape: (batch_size, num_labels)
    targets shape: (batch_size)
    """
    preds = preds.argmax(dim=1)
    acc = (preds == targets).float().mean()
    return acc
