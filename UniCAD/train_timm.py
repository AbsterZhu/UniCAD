import argparse
import logging
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import timm
from lora import LoRA_ViT_timm
from adapter import Adapter_ViT
from utils.dataloader_oai import kneeDataloader
from utils.dataloader_nih import nihDataloader
from utils.dataloader_mnist2d import mnist_2d_dataloader
from utils.dataloader_m3d import mnist_3d_dataloader
from utils.dataloader_inbreast import InbreastDataloader
from utils.result import ResultCLS,ResultMLS
from utils.utils import init
from utils.ViT_3D import ViT
import numpy as np
import random

from safetensors.torch import save_file
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
import types
from collections.abc import Sequence
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.layers import Conv, trunc_normal_
from monai.utils import deprecated_arg, ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_PATCH_EMBEDDING_TYPES = {"conv", "perceptron"}
SUPPORTED_POS_EMBEDDING_TYPES = {"none", "learnable", "sincos"}
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["http_proxy"]="http://127.0.0.1.1:7890"
# os.environ["https_proxy"]="http://127.0.0.1:7890"


#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

weightInfo={
            # "small":"WinKawaks/vit-small-patch16-224",
            "base":"vit_base_patch16_224.orig_in21k_ft_in1k",
            "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
            "base_sam":"vit_base_patch16_224.sam", # 1k
            "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
            "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
            "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
            "base_deit":"deit_base_distilled_patch16_224", # 1k
            "large":"google/vit-large-patch16-224",
            "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
            "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k", 
            "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
            "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
            "giant_clip":"vit_giant_patch14_clip_224.laion2b",
            "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
            }
    
    


def train(epoch,trainset):
    running_loss = 0.0
    this_lr = scheduler.get_last_lr()[0]
    net.train()
    for image, label in tqdm(trainset, ncols=60, desc="train", unit="b", leave=None):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast(enabled=True):
            pred = net.forward(image)
            loss = loss_func(pred, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss = running_loss + loss.item()
        scheduler.step()

    loss = running_loss / len(trainset)
    logging.info(f"\n\nEPOCH: {epoch}, LOSS : {loss:.3f}, LR: {this_lr:.2e}")
    return


@torch.no_grad()
def eval(epoch,testset,datatype='val'):
    result.init()
    net.eval()
    for image, label in tqdm(testset, ncols=60, desc=datatype, unit="b", leave=None):
        # if not torch.isnan(image).any():
        image, label = image.to(device), label.to(device)
        
        with autocast(enabled=True):
            # if image.dim()==5:
            #     pred = net.forward(image,cfg.train_type)#3d
            # else:
            pred = net.forward(image)
            result.eval(label, pred)
    result.print(epoch,datatype)
    return

def save_params(net,cpt_path):
    lora_and_patch_embedding_params = OrderedDict()

    for name, param in net.named_parameters():
        if 'lora' in name or 'patch_embedding' in name:
            lora_and_patch_embedding_params[name] = param.data

    save_file(lora_and_patch_embedding_params, ckpt_path)
    print(f"Saved LoRA and patch embedding parameters to {ckpt_path}")


from collections import OrderedDict

def load_pretrained_encoder(mae_model_path, vit_model):
    mae_state_dict = torch.load(mae_model_path)

    encoder_keys = [k for k in mae_state_dict if 'blocks' in k or 'patch_embedding' in k or k == 'norm.weight' or k == 'norm.bias']
    filtered_state_dict = {k: v for k, v in mae_state_dict.items() if k in encoder_keys}

    vit_state_dict = vit_model.state_dict()

    new_vit_state_dict = OrderedDict()
    for key in vit_state_dict:
        if key in filtered_state_dict:
            new_vit_state_dict[key] = filtered_state_dict[key]
        else:
            new_vit_state_dict[key] = vit_state_dict[key]

    vit_model.load_state_dict(new_vit_state_dict)
    return vit_model

class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4,
        >>>                     proj_type="conv", pos_embed_type="sincos")

    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            proj_type: patch embedding layer type.
            pos_embed_type: position embedding layer type.
            dropout_rate: fraction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.
        """

        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        self.proj_type = look_up_option(proj_type, SUPPORTED_PATCH_EMBEDDING_TYPES)
        self.pos_embed_type = look_up_option(pos_embed_type, SUPPORTED_POS_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.proj_type == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        if self.proj_type == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.proj_type == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))

        if self.pos_embed_type == "none":
            pass
        elif self.pos_embed_type == "learnable":
            trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.pos_embed_type == "sincos":
            grid_size = []
            for in_size, pa_size in zip(img_size, patch_size):
                grid_size.append(in_size // pa_size)

            self.position_embeddings = build_sincos_position_embedding(grid_size, hidden_size, spatial_dims)
        else:
            raise ValueError(f"pos_embed_type {self.pos_embed_type} not supported.")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embeddings(x)
        if self.proj_type == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        return embeddings
    
def _pos_embed_3d(self, x: torch.Tensor) -> torch.Tensor:
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    return x


if __name__ == "__main__":
    scaler = GradScaler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, default=16)
    parser.add_argument("-fold", type=int, default=0)
    parser.add_argument("-data_path",type=str, default='../data/NIH_X-ray/')
    parser.add_argument("-data_info",type=str,default='nih_split_712.json')
    parser.add_argument("-annotation",type=str,default='Data_Entry_2017_jpg.csv')
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-num_workers", type=int, default=1)
    parser.add_argument("-num_classes", "-nc", type=int, default=14)
    parser.add_argument("-train_type", "-tt", type=str, default="linear", help="lora, full, linear, adapter")
    parser.add_argument("-rank", "-r", type=int, default=4)
    parser.add_argument("-alpha", "-a", type=int, default=4)
    parser.add_argument("-vit", type=str, default="base")
    parser.add_argument("-data_size", type=float, default="1.0")
    cfg = parser.parse_args()
    ckpt_path = init(cfg)
    print(f"ckpt_path: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(cfg)

    if cfg.train_type=='resnet50':
        model=models.__dict__[cfg.train_type]()
        model.load_state_dict(torch.load('../preTrain/resnet50-19c8e357.pth'))
    else:
        if cfg.vit == "base":
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
        elif cfg.vit == "base_dino":
            model = timm.create_model(weightInfo["base_dino"], pretrained=True)
        elif cfg.vit == "base_sam":
            model = timm.create_model(weightInfo["base_sam"], pretrained=True)
        elif cfg.vit == "base_mill":
            model = timm.create_model(weightInfo["base_mill"], pretrained=True)
        elif cfg.vit == "base_beit":
            model = timm.create_model(weightInfo["base_beit"], pretrained=True)
        elif cfg.vit == "base_clip":
            model = timm.create_model(weightInfo["base_clip"], pretrained=True)
        elif cfg.vit == "base_deit":
            model = timm.create_model(weightInfo["base_deit"], pretrained=True)
        elif cfg.vit == "large_clip":
            model = timm.create_model(weightInfo["large_clip"], pretrained=True)
        elif cfg.vit == "large_beit":
            model = timm.create_model(weightInfo["large_beit"], pretrained=True)
        elif cfg.vit == "huge_clip":
            model = timm.create_model(weightInfo["huge_clip"], pretrained=True)
        elif cfg.vit == "giant_eva":
            model = timm.create_model(weightInfo["giant_eva"], pretrained=True)
        elif cfg.vit == "giant_clip":
            model = timm.create_model(weightInfo["giant_clip"], pretrained=True)
        elif cfg.vit == "giga_clip":
            model = timm.create_model(weightInfo["giga_clip"], pretrained=True)
        elif cfg.vit=="base_3d":
            model0 = ViT(
                    in_channels=1,
                    img_size=[16, 64, 64],
                    patch_size=[4, 16, 16],
                    hidden_size=768,
                    mlp_dim=3072,
                    num_layers=12,
                    num_heads=12,
                    pos_embed="perceptron",
                    dropout_rate=0,
                    spatial_dims=3,
                    classification=True,
                )
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            
            patch_embed_3d = PatchEmbeddingBlock(
                        in_channels=1,
                        img_size=[16, 64, 64],
                        patch_size=[4, 16, 16],
                        hidden_size=768,
                        num_heads=12,
                        pos_embed="perceptron",
                        spatial_dims=3,
                    )
            model.patch_embed = patch_embed_3d
            model._pos_embed = types.MethodType(_pos_embed_3d, model)
            
        elif "clip_3d" in cfg.vit:
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            if cfg.vit=="base_clip_3d":
                hidden_size = 768
                num_heads = 12
                model = timm.create_model(weightInfo["base_clip"], pretrained=True)
            if cfg.vit=="huge_clip_3d":
                hidden_size = 1280
                num_heads = 16
                model = timm.create_model(weightInfo["huge_clip"], pretrained=True)
            elif cfg.vit=="giant_clip_3d":
                hidden_size = 1408
                num_heads = 16
                model = timm.create_model(weightInfo["giant_clip"], pretrained=True)
            elif cfg.vit=="giga_clip_3d":
                hidden_size = 1664
                num_heads = 16
                model = timm.create_model(weightInfo["giga_clip"], pretrained=True)
            
            patch_embed_3d = PatchEmbeddingBlock(
                        in_channels=1,
                        img_size=[16, 64, 64],
                        patch_size=[4, 16, 16],
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        pos_embed="perceptron",
                        spatial_dims=3,
                    )
            model.patch_embed = patch_embed_3d
            model._pos_embed = types.MethodType(_pos_embed_3d, model)
                                            
        else:
            print("Wrong training type")
            exit()

    if cfg.train_type=='lora':
        lora_model = LoRA_ViT_timm(model, r=cfg.rank, alpha=cfg.alpha, num_classes=cfg.num_classes)
        num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        num_params1 = sum(p.numel() for p in lora_model.parameters() )

        print(f"trainable parameters: {num_params/2**20:.3f}M")
        print(f"parameters: {num_params1/2**20:.3f}M")
        net = lora_model.to(device)
    elif cfg.train_type == "adapter":
        adapter_model = Adapter_ViT(model, num_classes=cfg.num_classes)
        num_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = adapter_model.to(device)
    elif cfg.train_type == "full":
        model.reset_classifier(cfg.num_classes)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    elif cfg.train_type == "linear":
        model.reset_classifier(cfg.num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        num_params = sum(p.numel() for p in model.head.parameters())
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    elif cfg.train_type=='resnet50':
        infeature = model.fc.in_features
        model.fc = nn.Linear(infeature, cfg.num_classes)
        num_params = sum(p.numel() for p in model.fc.parameters())
        print(f"trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    elif cfg.train_type=='lora3d':
        model = LoRA_ViT_timm(model, r=cfg.rank, alpha=cfg.alpha, num_classes=cfg.num_classes)
        for param in model.lora_vit.patch_embed.parameters():
            param.requires_grad = True
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params/2**20:.3f}M")
        net = model.to(device)
    else:
        print("Wrong training type")
        exit()
    net = torch.nn.DataParallel(net)
    if cfg.data_path.split('/')[-1] == "OAI-train":#路径改成取最后一个
        trainset, valset, testset = kneeDataloader(cfg)
    elif "NIH_X-ray" in cfg.data_path:
        trainset, valset, testset=nihDataloader(cfg)
    elif 'mnist3d' in cfg.data_path:
        trainset, valset, testset=mnist_3d_dataloader(cfg)  
    elif 'mnist' in cfg.data_path:
        trainset, valset, testset=mnist_2d_dataloader(cfg)
    elif 'INBreast'in cfg.data_path:
        trainset, valset, testset=InbreastDataloader(cfg,0)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, 1e-6)
    if "NIH_X-ray" in cfg.data_path:
        loss_func = nn.BCEWithLogitsLoss().to(device)
        result = ResultMLS(cfg.num_classes)
    else:
        loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        result = ResultCLS(cfg.num_classes)

    for epoch in range(1, cfg.epochs+1):
        train(epoch,trainset)
        if epoch%1==0:
            eval(epoch,valset,datatype='val')
            if result.best_epoch == result.epoch:
                if cfg.train_type == "lora":
                    net.module.save_lora_parameters(ckpt_path.replace(".pt", ".safetensors"))
                elif cfg.train_type == "lora3d":
                    net.module.save_lora_parameters(ckpt_path.replace(".pt", ".safetensors"),dim='3d')
                else:
                    torch.save(net.state_dict(), ckpt_path.replace(".pt", "_best.pt"))
                eval(epoch,testset,datatype='test')
                logging.info(f"BEST VAL: {result.best_val_result:.3f}, TEST: {result.test_auc:.3f}, EPOCH: {(result.best_epoch):3}")
