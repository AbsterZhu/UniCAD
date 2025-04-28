import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image,ImageSequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
import monai.transforms as mtf
import random
import codecs
import csv

import matplotlib.pyplot as plt
def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class MNIST_3D_Dataset(Dataset):
    def __init__(self, data_type="train", fold_idx=0, data_path="bloodmnist_224", data_size=1.0, vit_type="base"):
        self.cases = []
        self.labels = []
        self.data_type = data_type
        self.vit_type = vit_type
        
        self.root_path = "/root_path/to/your/dataset"#change path to your dataset
        self.img_path = os.path.join(self.root_path, data_path)
        self.csv_path = os.path.join(self.root_path, f"{data_path}.csv")
        self.resize= mtf.Compose([
            mtf.Resize(spatial_size=[16, 64, 64], mode="bilinear")
            ])

        with open(self.csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if self.data_type.upper() in row[0]:
                    self.cases.append(os.path.join(self.img_path, row[1]))
                    self.labels.append(int(row[2]))
                    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        image = Image.open(self.cases[idx])
        frame_list=[]
        for frame in ImageSequence.Iterator(image):
            try:
                frame = frame.convert("L")
            except ValueError:
                # 处理没有调色板的图像
                frame = frame.convert("RGB").convert("L")
        
            frame_array = np.array(frame) / 255.0
            frame_list.append(frame_array)
        frame_array=np.stack(frame_list,axis=0)
        frame_array=frame_array[np.newaxis,:]
                
        frame_array = self.resize(frame_array)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return frame_array, label
    
                    
def mnist_3d_dataloader(cfg):
    train_set = DataLoader(
        MNIST_3D_Dataset(data_type="train", fold_idx=cfg.fold, data_path=cfg.data_path, vit_type=cfg.vit),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    
    val_set = DataLoader(
        MNIST_3D_Dataset(data_type="val", fold_idx=cfg.fold, data_path=cfg.data_path, vit_type=cfg.vit),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    test_set = DataLoader(
        MNIST_3D_Dataset(data_type="test", fold_idx=cfg.fold, data_path=cfg.data_path, vit_type=cfg.vit),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_set, val_set, test_set


# if __name__=='__main__':
#     PATH="/public_bme/data/medmnist/vesselmnist3d_64/test100_0.gif"
#     image = Image.open(PATH)

#     index = 0
#     frame_list=[]
#     for frame in ImageSequence.Iterator(image):
#         print(np.array(frame).shape)
#         frame_list.append(np.array(frame.convert("L"))/255.0)
#         index += 1
#     frame_array=np.stack(frame_list,axis=0)
#     frame_array=frame_array[np.newaxis,:]
    
#     print(frame_array.shape)
#     transform = mtf.Compose([
#     mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
# ])
#     image = transform(frame_array)
#     print(image.shape)
    
