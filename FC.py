import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from resnet import resnet18, resnet50, resnet101
from swin_transformer import swin_transformer_base
from basic_block import *

class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)

        f_u = self.relu(f_u)

        return f_u

class MEFARG(nn.Module):
    def __init__(self, num_classes=13, backbone='resnet50'):
        super(MEFARG, self).__init__()
        if 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18(pretrained=True)
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50(pretrained=True)
        
            self.in_channels = self.backbone.fc.weight.shape[1]  # backbone.fc.weight.shape = [1000,512]

            self.backbone.fc = None
        elif 'transformer' in backbone:
            if 'transformer' in backbone:
                if backbone == 'swin_transformer_tiny':
                    self.backbone = swin_transformer_tiny()
                elif backbone == 'swin_transformer_small':
                    self.backbone = swin_transformer_small()
                else:
                    self.backbone = swin_transformer_base()
                self.in_channels = self.backbone.num_features
                self.backbone.head = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        
        self.head = Head(self.in_channels, num_classes)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)

        cl = self.head(x)
        return cl










