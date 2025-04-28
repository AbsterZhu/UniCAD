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
import codecs
import csv
def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MNIST_2D_Dataset(Dataset):
    def __init__(self, data_type="train", fold_idx=0, data_path="bloodmnist_224", data_size=1.0, vit_type="base"):
        self.cases = []
        self.labels = []
        self.data_type = data_type
        self.vit_type = vit_type
        self.random_flip = RandomHorizontalFlip(p=0.5)
        self.resize = Resize([224, 224])
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.root_path =  "/root_path/to/your/dataset"#change path to your dataset
        self.img_path = os.path.join(self.root_path, data_path)
        self.csv_path = os.path.join(self.root_path, f"{data_path}.csv")

        with open(self.csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if self.data_type.upper() in row[0]:
                    self.cases.append(os.path.join(self.img_path, row[1]))
                    self.labels.append(int(row[2]))
                    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.cases[idx]).convert("RGB")).astype(np.float32) / 255.0
        image = rearrange(torch.tensor(image, dtype=torch.float32), 'h w c -> c h w')
        image = self.resize(image)
        image = self.normalize(image)
        if self.data_type == "train":
            image = self.random_flip(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label
                    
                    
def mnist_2d_dataloader(cfg):
    train_set = DataLoader(
        MNIST_2D_Dataset(data_type="train", fold_idx=cfg.fold, data_path=cfg.data_path, vit_type=cfg.vit),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    
    val_set = DataLoader(
        MNIST_2D_Dataset(data_type="val", fold_idx=cfg.fold, data_path=cfg.data_path, vit_type=cfg.vit),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    test_set = DataLoader(
        MNIST_2D_Dataset(data_type="test", fold_idx=cfg.fold, data_path=cfg.data_path, vit_type=cfg.vit),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_set, val_set, test_set