# UniCAD: Efficient and Extendable Architecture for Multi-Task CAD System in Clinical Scenarios
The official implementation for "UniCAD: Efficient and Extendable Architecture for Multi-Task CAD System in Clinical Scenarios".

## Useful links

<div align="center">
    <a href="https://mii-laboratory.github.io/UniCAD/" class="button"><b>[Homepage]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

## Feature
- [x] Release training code. 
- [ ] Release multi-task inference code.

## Install 
```bash
conda create -n muc python=3.9
conda activate unicad
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install timm
```

## Preparation
- Download [MedMNIST dataset](https://medmnist.com/)
- Change the dataset path in utils/dataloader_mnist.py

## Training
### For training on MedMNIST 2D datasets (example: dermamnist_224 dataset)
```
cd UniCAD
python train_timm.py  -bs 32 -data_path "dermamnist_224" -data_info "dermamnist_224" -lr 3e-4 -epochs 100 -num_classes 7 -train_type "lora" -rank 4 -vit "base"
```
### For training on MedMNIST 2D datasets (example: adrenalmnist3d_64 dataset)
```
cd UniCAD
python train_timm.py  -bs 32 -data_path "adrenalmnist3d_64" -data_info "adrenalmnist3d_64" -lr 3e-4 -epochs 100 -num_classes 2 -train_type "lora3d" -rank 4 -vit "base_3d"
```

## References
- [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT)

