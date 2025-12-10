![Validation Accuracy](images/val_acc.jpg)
GapAwareNet

A Modern CNN That Bridges Classic Deep Learning Gaps

GapAwareNet is a hybrid deep learning architecture that combines ConvNeXt, Vision Transformers (ViT), and Spatial Transformers (STN). It improves global reasoning, spatial awareness, robustness, and generalization for computer vision tasks.

Key Features

Limited global context is addressed using lightweight multi-head self-attention.

Spatial information loss is reduced using dilated depthwise convolutions and optional skip connections.

Sensitivity to rotation and scaling is handled using Spatial Transformer Networks and BlurPool.

Texture bias over shape is reduced using large depthwise convolutions and strong augmentations.

Fragility to adversarial inputs is handled using FGSM-based adversarial support.

Overfitting and data hunger are reduced using MixUp, CutMix, label smoothing, and stochastic depth.

Lack of channel focus is addressed using squeeze-and-excitation attention.

Task rigidity is removed using a classifier that supports CAM and optional segmentation head.

Architecture Overview

Input
Spatial Transformer Network
CoordConv2D Stem
Stage 1 (ConvNeXt blocks)
Stage 2 (ConvNeXt with self-attention)
Stage 3 (ConvNeXt with self-attention)
Stage 4 (ConvNeXt blocks)
Adaptive Pool and 1x1 convolution for classification

Each stage includes anti-aliased downsampling, ConvNeXt-style depthwise convolutions, optional self-attention, and support for dilated convolutions.

Components

Spatial Transformer Network learns affine transforms for robustness.
CoordConv2D adds coordinate information.
ConvNeXt block includes depthwise 7x7 convolution, MLP, and squeeze-and-excitation attention.
Self-attention provides lightweight contextual reasoning.
BlurPool improves downsampling smoothness.
MixUp and CutMix improve generalization.
Label smoothing prevents overconfident predictions.
FGSM hook provides adversarial robustness.
CAM utility enables visual explanation.

Requirements

Python 3.8 or above
PyTorch 2.0 or above
torchvision 0.15 or above
CUDA GPU recommended

Install:
pip install torch torchvision

Quick Start

Clone repository:
git clone https://github.com/yourusername/GapAwareNet.git

cd GapAwareNet

Train on CIFAR-10:
python gap_aware_net.py --dataset cifar10 --epochs 5 --batch-size 128 --lr 1e-3

Command Line Arguments

--dataset : dataset name
--img-size : input image size
--epochs : number of epochs
--batch-size : batch size
--lr : learning rate
--mixup-alpha : MixUp strength
--cutmix-alpha : CutMix strength
--label-smoothing : smoothing factor
--adv : enable adversarial support
--adv-every : FGSM frequency
--eps : FGSM perturbation level
--out : output model path

Evaluating a Saved Model

import torch
from gap_aware_net import GapAwareNet

model = GapAwareNet(num_classes=10)
model.load_state_dict(torch.load('gap_aware_net.pt', map_location='cpu'))
model.eval()

Using Class Activation Map

from gap_aware_net import class_activation_map
import matplotlib.pyplot as plt

with torch.no_grad():
logits, feats = model(images)
cam = class_activation_map(feats, model.head, class_idx=3)
plt.imshow(cam[0].cpu(), cmap='jet')
plt.show()

Extending to Segmentation or Detection

feats = model.forward_features(x)

You can attach segmentation heads, detection heads, or CAM modules.

Performance Summary

CIFAR-10 accuracy: around 91 to 93 percent
CIFAR-100 accuracy: around 74 to 78 percent
ImageNet-mini accuracy: around 68 to 72 percent
