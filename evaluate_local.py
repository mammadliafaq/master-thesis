import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder

from dataset import ShopeeDataset
from models.image_model import ImageModel
from transforms import get_valid_transforms
from utils.eval_utils import (generate_test_features, plot_threshold,
                              predict_img)
from utils.utils import convert_dict_to_tuple, set_seed


def evaluate(args: argparse.Namespace) -> None:
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

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
    valid_dataset = ShopeeDataset(
        csv=valid,
        transforms=get_valid_transforms(config),
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.dataset.num_workers,
    )

    # getting model and checkpoint
    print("Creating model and loading checkpoint")
    model_params = {
        "n_classes": config.dataset.num_of_classes,
        "model_name": config.model.model_name,
        "pretrained": config.model.pretrained,
        "use_fc": config.model.use_fc,
        "fc_dim": config.model.fc_dim,
        "dropout": config.model.dropout,
        "loss_module": config.model.loss_module,
        "s": config.model.s,
        "margin": config.model.margin,
        "ls_eps": config.model.ls_eps,
        "theta_zero": config.model.theta_zero,
    }
    model = ImageModel(**model_params)

    checkpoint = torch.load(args.weights, map_location="cuda")

    model.load_state_dict(checkpoint)
    print("Weights have been loaded successfully.")
    model.to(device)

    print("Generating features for the validation set to evaluate f1 score..")
    image_embeddings = generate_test_features(config, model, valid_loader, device)

    if config.search_for_threshold:
        print("Searching for the best threshold....")
        threshold_range = np.arange(0.1, 1.0, 0.01)
        scores = []

        for threshold in threshold_range:
            print(
                f"*************************************** Threshold={threshold} ***************************************"
            )
            f1_score, _ = predict_img(valid, image_embeddings, threshold=threshold)
            scores.append(f1_score)
            print(f"F1 score for the threshold={threshold} is {f1_score}")

        print("Saving the threshold plot...")
        path_to_plot = os.path.join(
            config.outdir, config.exp_name, "threshold_plot.png"
        )
        plot_threshold(threshold_range, scores, path_to_plot)

        threshold_dict = {t: f1 for t, f1 in zip(threshold_range, np.array(scores))}
        sorted_threshold_dict = dict(
            sorted(threshold_dict.items(), key=lambda item: item[1])
        )
        sorted_list_items = list(sorted_threshold_dict.items())

        best_threshold = sorted_list_items[-1][0]
        best_f1_score = sorted_list_items[-1][1]

        print(f"The F1 score = {best_f1_score} with best threshold = {best_threshold}")
    else:
        f1_score, output_df = predict_img(
            valid, image_embeddings, threshold=config.threshold
        )
        print(output_df.head())
        submit_df = output_df[["posting_id", "pred_imgonly", "target"]].copy()
        submit_df.rename(columns={"pred_imgonly": "matches"}, inplace=True)
        submit_df.to_csv(os.path.join(config.outdir, config.exp_name, "submission.csv"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="Path to config file.")
    parser.add_argument("--weights", type=str, default="", help="Path to weights.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_arguments(sys.argv[1:]))
