import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
import random
import json

disease={"Atelectasis":0,
        "Cardiomegaly":1,
        "Effusion":2,
        "Infiltration":3,
        "Mass":4,
        "Nodule":5,
        "Pneumonia":6,
        "Pneumothorax":7,
        "Consolidation":8,
        "Edema":9,
        "Emphysema":10,
        "Fibrosis":11,
        "Pleural_Thickening":12,
        "Hernia":13,
}

class GraphDataset(Dataset):
    def __init__(self, data_type="train", fold_idx=0, data_path="ChinaSet_AllFiles"):
        self.data_path = data_path
        self.data_type = data_type
        self.name_list = json.load(open(os.path.join(data_path, 'nih_split_712.json')))
        self.image_path = os.path.join(data_path, 'images')
        self.df = pd.read_csv(os.path.join(data_path, "Data_Entry_2017_jpg.csv"))
        
        self.resize = Resize([224, 224])
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.random_flip = RandomHorizontalFlip(p=0.5)
    
        if data_type == "train":
            self.cases = self.name_list["train"]
        elif data_type == "test":
            self.cases = self.name_list["test"]
        elif data_type == "val":
            self.cases = self.name_list["val"]
        else:
            print("Dataset type error")
            exit()
            
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        
        
        file_name = self.cases[idx]
        image = np.array(Image.open(os.path.join(self.image_path, file_name)).convert('RGB')).astype(np.float32) / 255.0
        image = rearrange(torch.tensor(image, dtype=torch.float32), 'h w c -> c h w')
        image = self.resize(image)
        image = self.normalize(image)
        if self.data_type == "train":
            image = self.random_flip(image)
        
        findings = self.df.loc[self.df['Image Index'].values==file_name]['Finding Labels'].values[0].split("|")
        gt = np.zeros([len(disease)], dtype=np.int64)
        if findings[0] != "No Finding":
            gt[list(map(lambda x: disease[x], findings))]=1
        
        gt=torch.tensor(gt,dtype=torch.float32)
        
        return image, gt
    
    
def nihDataloader(cfg):
    train_set = DataLoader(
        GraphDataset(data_type="train", fold_idx=cfg.fold, data_path=cfg.data_path),
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
        drop_last=True,
    )
    
    test_set = DataLoader(
        GraphDataset(data_type="test", fold_idx=cfg.fold, data_path=cfg.data_path),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    
    return train_set, val_set, test_set

# if __name__=="__main__":
#     prev_case=None
#     dataInfo={}
#     testset=[]
#     with open('../data/NIH_X-ray/test_list_jpg.txt','r') as f:
#         content=f.readlines()
#         for c in content:
#             testset.append(c.strip('\n'))
    
#     trainset=[]
#     valset=[]
#     train_ratio=7/8
#     with open('../data/NIH_X-ray/train_val_list_jpg.txt','r') as f:
#         train_content=f.readlines()
#         trainNum=int(len(train_content)*train_ratio)
#         for i in range(0,trainNum):
#             trainset.append(train_content[i].strip('\n'))
#         for i in range(trainNum,len(train_content)):
#             valset.append(train_content[i].strip('\n'))
#         # for c in content:
#         #     testset.append(c.strip('\n'))
#     # dataInfo['test']=testset
#     dataInfo['meta']={'trainSize':len(trainset),'valSize':len(valset),'testSize':len(testset)}
#     dataInfo['train']=trainset
#     dataInfo['val']=valset
#     dataInfo['test']=testset
    
#     with open('nih_split_712.json', 'w') as json_file:
#         json.dump(dataInfo, json_file,indent = 4)

