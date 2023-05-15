import os.path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1


def generate_test_features(config, model, dataloader, device):
    model.eval()
    bar = tqdm(dataloader)

    feature_dim = config.model.fc_dim

    embeddings = np.empty((0, feature_dim), dtype="float32")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(bar):
            images = images.to(device)
            features = model.extract_features(images)
            features_normalized = F.normalize(features, dim=1)
            embeddings = np.append(
                embeddings, features_normalized.cpu().detach().numpy(), axis=0
            )
    return embeddings


def predict_img(df, embeddings, topk=50, threshold=0.63):
    df_copy = df.copy()
    N, D = embeddings.shape
    cpu_index = faiss.IndexFlatL2(D)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(embeddings)
    cluster_distance, cluster_index = gpu_index.search(x=embeddings, k=topk)

    # Make predictions
    df_copy["pred_images"] = ""
    pred = []
    for k in range(embeddings.shape[0]):
        idx = np.where(cluster_distance[k,] < threshold)[0]
        ids = cluster_index[k, idx]
        posting_ids = df_copy["posting_id"].iloc[ids].values
        pred.append(posting_ids)
    df_copy["pred_images"] = pred

    # Create target
    tmp = df_copy.groupby("label_group").posting_id.agg("unique").to_dict()
    df_copy["target"] = df_copy.label_group.map(tmp)
    df_copy["target"] = df_copy["target"].apply(lambda x: " ".join(x))
    # Calculate metrics
    df_copy["pred_imgonly"] = df_copy.pred_images.apply(lambda x: " ".join(x))
    df_copy["f1_img"] = f1_score(df_copy["target"], df_copy["pred_imgonly"])
    score = df_copy["f1_img"].mean()
    return score, df_copy


def plot_threshold(threshold, f1_scores, path_to_save):
    plt.plot(threshold, np.array(f1_scores), "ob")

    #### Lable and Grid ####################
    plt.xlabel("Threshold")  # x label
    plt.ylabel("F1 scores")  # y label
    plt.grid()  # show grid
    plt.savefig(path_to_save)
