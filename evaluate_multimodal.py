import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder

from dataset import ShopeeMultimodalDataset
from models.image_model import ShopeeImageModel
from models.multimodal_model import ShopeeMultiModel
from models.text_model import ShopeeTextModel
from transforms import get_valid_transforms
from utils.eval_utils import (generate_fused_features, get_multimodal_predictions,
                              plot_threshold)
from utils.utils import convert_dict_to_tuple, set_seed

import transformers


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

    print("Setting up the tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.text_model.model_name)

    # Defining DataSet
    valid_dataset = ShopeeMultimodalDataset(
        csv=valid,
        image_transforms=get_valid_transforms(config),
        tokenizer=tokenizer
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.dataset.num_workers,
    )

    # Setting up the image model and checkpoint
    print("Creating model and loading checkpoint")
    # Define image model
    image_model_params = {
        "n_classes": config.dataset.num_of_classes,
        "model_name": config.image_model.model_name,
        "pretrained": config.image_model.pretrained,
        "use_fc": config.image_model.use_fc,
        "fc_dim": config.image_model.fc_dim,
        "dropout": config.image_model.dropout,
    }
    image_model = ShopeeImageModel(**image_model_params, device=device)
    image_checkpoint = torch.load(args.image_weights, map_location="cuda")
    image_model.load_state_dict(image_checkpoint)
    print("Image model weights have been loaded successfully.")
    image_model.to(device)

    # Define text model
    text_model_params = {
        "n_classes": config.dataset.num_of_classes,
        "model_name": config.text_model.model_name,
        "use_fc": config.text_model.use_fc,
        "fc_dim": config.text_model.fc_dim,
        "dropout": config.text_model.dropout,
    }
    text_model = ShopeeTextModel(**text_model_params, device=device)
    if args.text_weights:
        text_checkpoint = torch.load(args.text_weights, map_location="cuda")

        text_model.load_state_dict(text_checkpoint)
        print("Text model weights have been loaded successfully.")
    else:
        print("Use pretrained weights.")
    text_model.to(device)

    # Define multimodal model
    multi_model_params = {
        "n_classes": config.dataset.num_of_classes,
        "use_fc": config.multi.use_fc,
        "fc_dim": config.multi.fc_dim,
        "dropout": config.multi.dropout,
        "loss_module": config.multi.loss_module,
        "s": config.head.s,
        "margin": config.head.margin,
        "ls_eps": config.head.ls_eps,
        "theta_zero": config.head.theta_zero,
    }
    num_image_features = image_model.final_in_features
    num_text_features = text_model.final_in_features

    print(f"Number of image features before fusing = {num_image_features}")
    print(f"Number of text features before fusing = {num_text_features}")

    multi_model = ShopeeMultiModel(
        **multi_model_params,
        num_image_features=num_image_features,
        num_text_features=num_text_features,
        device=device,
    )
    multi_checkpoint = torch.load(args.multi_weights, map_location="cuda")
    multi_model.load_state_dict(multi_checkpoint)
    print("Fusing model weights have been loaded successfully.")

    multi_model.to(device)

    print("Generating features for the validation set to evaluate f1 score..")
    multi_embeddings = generate_fused_features(image_model, text_model, multi_model, valid_loader, device)

    print(multi_embeddings.shape)

    if config.search_for_threshold:
        print("Searching for the best threshold....")
        threshold_range = np.arange(0.1, 1.0, 0.01)
        scores = []

        for threshold in threshold_range:
            print(
                f"*************************************** Threshold={threshold} ***************************************"
            )
            f1_score, _ = get_multimodal_predictions(
                valid, multi_embeddings, threshold=threshold
            )
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
        f1_score, output_df = get_multimodal_predictions(
            valid, multi_embeddings, threshold=config.threshold
        )
        print(output_df.head())
        submit_df = output_df[["posting_id", "pred_multi_only", "target"]].copy()
        submit_df.rename(columns={"pred_multi_only": "matches"}, inplace=True)
        submit_df.to_csv(os.path.join(config.outdir, config.exp_name, "submission.csv"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="Path to config file.")
    parser.add_argument("--image_weights", type=str, default="", help="Path to image weights.")
    parser.add_argument("--text_weights", type=str, default="", help="Path to text weights.")
    parser.add_argument("--multi_weights", type=str, default="", help="Path to multimodal weights.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_arguments(sys.argv[1:]))
