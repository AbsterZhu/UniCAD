import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
import random


class GraphDataset(Dataset):
    def __init__(self, data_type="train", fold_idx=0, data_path="OAI-train", data_size=1.0):
        cases = []
        labels = []
        self.data_type=data_type
        for grade in range(5):
            _cases = os.listdir(f"{data_path}/{grade}")
            _cases = [f"{data_path}/{grade}/{_}" for _ in _cases if ".png" in _]
            _labels = [grade for _ in _cases]
            cases = cases + _cases
            labels = labels + _labels

        train_cases, test_cases, train_labels, test_labels = train_test_split(cases, labels, test_size=0.2, shuffle=True, random_state=42)
        test_cases, val_cases, test_labels, val_labels = train_test_split(test_cases, test_labels, test_size=0.5, shuffle=True, random_state=42)

        
        if self.data_type == "train":
            assert ((data_size > 0) and (data_size <= 1.0))
            temp = list(zip(train_cases, train_labels))
            random.shuffle(temp)
            train_cases[:], train_labels[:] = zip(*temp)
            self.cases = np.array(train_cases[:int(len(train_cases) * data_size)])
            self.labels = np.array(train_labels[:int(len(train_cases) * data_size)])
        elif self.data_type == "test":
            self.cases = np.array(test_cases)
            self.labels = np.array(test_labels)
        elif self.data_type == "val":
            self.cases = np.array(val_cases)
            self.labels = np.array(val_labels)
        else:
            print("Dataset type error")
            exit()

    def __len__(self):
        # return 100
        return len(self.cases)

    def __getitem__(self, idx):
        resize = Resize([224, 224])
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        random_flip = RandomHorizontalFlip(p=0.5)
        image = np.array(Image.open(self.cases[idx]).convert("RGB")).astype(np.float32) / 255.0
        image = rearrange(torch.tensor(image, dtype=torch.float32), 'h w c -> c h w')

        image = resize(image)
        image = normalize(image)
        
        if self.data_type=='train':
            image = random_flip(image)  
        
        label = int(self.labels[idx])
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def _kneeDataloader(bs):
    train_set = DataLoader(
        GraphDataset(data_type="train", fold_idx=0),
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    test_set = DataLoader(
        GraphDataset(data_type="test", fold_idx=0),
        batch_size=bs,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    return train_set, test_set


def kneeDataloader(cfg):
    train_set = DataLoader(
        GraphDataset(data_type="train", fold_idx=cfg.fold, data_path=cfg.data_path, data_size=cfg.data_size),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    
    val_set = DataLoader(
        GraphDataset(data_type="val", fold_idx=cfg.fold, data_path=cfg.data_path),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    test_set = DataLoader(
        GraphDataset(data_type="test", fold_idx=cfg.fold, data_path=cfg.data_path),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_set, val_set, test_set

