import cv2
import torch
from torch.utils.data import Dataset


class ShopeeImageDataset(Dataset):
    def __init__(self, csv, transforms=None):
        self.csv = csv.reset_index()
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        text = row.title

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        return image, torch.tensor(row.label_group)


class ShopeeTextDataset(Dataset):
    def __init__(self, csv, tokenizer):
        self.csv = csv.reset_index()
        self.tokenizer = tokenizer

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        text = row.title

        text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]

        return input_ids, attention_mask, torch.tensor(row.label_group)
