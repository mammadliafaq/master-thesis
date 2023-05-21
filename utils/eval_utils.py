import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

import gc


def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1


def plot_threshold(threshold, f1_scores, path_to_save):
    plt.plot(threshold, np.array(f1_scores), "ob")

    plt.xlabel("Threshold")  # x label
    plt.ylabel("F1 scores")  # y label
    plt.grid()  # show grid
    plt.savefig(path_to_save)


def generate_image_features(config, model, dataloader, device):
    model.eval()
    bar = tqdm(dataloader)

    feature_dim = config.image_model.fc_dim

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


def generate_text_features(model, dataloader, device):
    embeds = []

    model.eval()
    bar = tqdm(dataloader)

    with torch.no_grad():
        for input_ids, attention_mask, _ in bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = model.extract_features(input_ids, attention_mask)
            features_normalized = F.normalize(features, dim=1)
            text_features = features_normalized.detach().cpu().numpy()
            embeds.append(text_features)

    text_embeddings = np.concatenate(embeds)
    print(f'Our text embeddings shape is {text_embeddings.shape}')
    return text_embeddings


def get_image_predictions(df, image_embeddings, topk=50, threshold=0.63):
    df_copy = df.copy()
    N, D = image_embeddings.shape
    cpu_index = faiss.IndexFlatL2(D)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(image_embeddings)
    cluster_distance, cluster_index = gpu_index.search(x=image_embeddings, k=topk)

    # Make predictions
    df_copy["pred_images"] = ""
    image_predictions = []
    for k in range(image_embeddings.shape[0]):
        idx = np.where(cluster_distance[k,] < threshold)[0]
        ids = cluster_index[k, idx]
        posting_ids = df_copy["posting_id"].iloc[ids].values
        image_predictions.append(posting_ids)
    df_copy["pred_images"] = image_predictions

    # Create target
    tmp = df_copy.groupby("label_group").posting_id.agg("unique").to_dict()
    df_copy["target"] = df_copy.label_group.map(tmp)
    df_copy["target"] = df_copy["target"].apply(lambda x: " ".join(x))

    # Calculate metrics
    df_copy["pred_img_only"] = df_copy.pred_images.apply(lambda x: " ".join(x))
    df_copy["f1_img"] = f1_score(df_copy["target"], df_copy["pred_img_only"])
    score = df_copy["f1_img"].mean()
    return score, df_copy


def get_text_predictions(df, image_embeddings, topk=50, threshold=0.63):
    df_copy = df.copy()
    N, D = image_embeddings.shape
    cpu_index = faiss.IndexFlatL2(D)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(image_embeddings)
    cluster_distance, cluster_index = gpu_index.search(x=image_embeddings, k=topk)

    # Make predictions
    df_copy["pred_text"] = ""
    image_predictions = []
    for k in range(image_embeddings.shape[0]):
        idx = np.where(cluster_distance[k,] < threshold)[0]
        ids = cluster_index[k, idx]
        posting_ids = df_copy["posting_id"].iloc[ids].values
        image_predictions.append(posting_ids)
    df_copy["pred_text"] = image_predictions

    # Create target
    tmp = df_copy.groupby("label_group").posting_id.agg("unique").to_dict()
    df_copy["target"] = df_copy.label_group.map(tmp)
    df_copy["target"] = df_copy["target"].apply(lambda x: " ".join(x))

    # Calculate metrics
    df_copy["pred_text_only"] = df_copy.pred_text.apply(lambda x: " ".join(x))
    df_copy["f1_text"] = f1_score(df_copy["target"], df_copy["pred_text_only"])
    score = df_copy["f1_text"].mean()
    return score, df_copy


def get_tfidf_predictions_torch(df, device, max_features=25_000, th=0.75):
    model = TfidfVectorizer(
        stop_words="english", binary=True, max_features=max_features
    )
    text_embeddings = model.fit_transform(df["title"])

    text_embeddings = text_embeddings.toarray().astype(np.float16)
    print(text_embeddings.shape)
    text_embeddings = torch.from_numpy(text_embeddings).to(device)  # .half()

    CHUNK = 1024
    CTS = len(df) // CHUNK
    if (len(df) % CHUNK) != 0:
        CTS += 1

    preds = []
    indexes = []
    for j in tqdm(range(CTS)):
        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(df))
        cts = torch.matmul(text_embeddings, text_embeddings[a:b].T).T
        for k in range(b - a):
            IDX = torch.where(cts[k,] > th)[0].cpu().numpy()
            o = df.iloc[IDX].posting_id.values
            preds.append(o)
            indexes.append(IDX)

    gc.collect()
    return preds


def get_tfidf_predictions(df, cluster_distance, cluster_index, threshold=0.63):
    df_copy = df.copy()

    # Make predictions
    df_copy["pred_text"] = ""
    tf_idf_predictions = []
    for k in range(cluster_distance.shape[0]):
        idx = np.where(cluster_distance[k,] < threshold)[0]
        ids = cluster_index[k, idx]
        posting_ids = df_copy["posting_id"].iloc[ids].values
        tf_idf_predictions.append(posting_ids)
    df_copy["pred_text"] = tf_idf_predictions

    # Create target
    tmp = df_copy.groupby("label_group").posting_id.agg("unique").to_dict()
    df_copy["target"] = df_copy.label_group.map(tmp)
    df_copy["target"] = df_copy["target"].apply(lambda x: " ".join(x))

    # Calculate metrics
    df_copy["pred_text_only"] = df_copy.pred_text.apply(lambda x: " ".join(x))
    df_copy["f1_text"] = f1_score(df_copy["target"], df_copy["pred_text_only"])
    score = df_copy["f1_text"].mean()
    return score, df_copy
