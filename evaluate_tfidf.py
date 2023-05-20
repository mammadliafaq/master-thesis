import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.eval_utils import f1_score, get_tfidf_predictions_torch, get_tfidf_predictions, plot_threshold
from utils.utils import convert_dict_to_tuple, set_seed

import faiss


def read_dataset(config):
    df = pd.read_csv(config.dataset.path_to_folds)
    return df


def combine_predictions(row):
    x = np.concatenate([row["text_predictions"], row["phash"]])
    return " ".join(np.unique(x))


def evaluate_torch(args: argparse.Namespace) -> None:
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    # Set seed
    set_seed(config.seed)

    # Defining Device
    device_id = config.gpu_id
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("Selected device: ", device)

    df = read_dataset(config)
    print(df.head())

    text_predictions = get_tfidf_predictions_torch(df, device, max_features=25_000)

    phash = df.groupby("image_phash").posting_id.agg("unique").to_dict()
    df["phash"] = df.image_phash.map(phash)

    # Create target
    tmp = df.groupby("label_group").posting_id.agg("unique").to_dict()
    df["target"] = df.label_group.map(tmp)
    df["target"] = df["target"].apply(lambda x: " ".join(x))

    df["text_predictions"] = text_predictions
    df["text_predictions_str"] = df["text_predictions"].apply(lambda x: " ".join(x))
    df["f1_img"] = f1_score(df["target"], df["text_predictions_str"])
    score = df["f1_img"].mean()

    df["matches"] = df.apply(combine_predictions, axis=1)
    df[["posting_id", "matches"]].to_csv("submission.csv", index=False)
    print(df.head())
    return score


def evaluate_faiss(args: argparse.Namespace) -> None:
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    # Set seed
    set_seed(config.seed)

    # Defining Device
    device_id = config.gpu_id
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("Selected device: ", device)

    df = read_dataset(config)
    print(df.head())

    model = TfidfVectorizer(
        stop_words="english", binary=True, max_features=config.tf_idf.max_features
    )
    text_embeddings = model.fit_transform(df["title"])

    text_embeddings = text_embeddings.toarray().astype(np.float16)
    print(f"Evaluated text embeddings using TF-IDF! Shape={text_embeddings.shape}")

    print("Applying FAISS to calculate L2 distances")
    N, D = text_embeddings.shape
    cpu_index = faiss.IndexFlatL2(D)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(text_embeddings)
    cluster_distance, cluster_index = gpu_index.search(x=text_embeddings, k=config.topk)

    if config.search_for_threshold:
        print("Searching for the best threshold....")
        threshold_range = np.arange(0.1, 1.5, 0.01)
        scores = []

        for threshold in threshold_range:
            print(
                f"*************************************** Threshold={threshold} ***************************************"
            )
            f1_score, _ = get_tfidf_predictions(df, cluster_distance, cluster_index, threshold=threshold)
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
        f1_score, output_df = get_tfidf_predictions(
            df, cluster_distance, cluster_index, threshold=config.threshold
        )
        print(output_df.head())
        submit_df = output_df[["posting_id", "pred_text_only", "target"]].copy()
        submit_df.rename(columns={"pred_text_only": "matches"}, inplace=True)
        submit_df.to_csv(os.path.join(config.outdir, config.exp_name, "submission.csv"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="Path to config file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate_faiss(parse_arguments(sys.argv[1:]))
