import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from FC import MEFARG
from MLPProcess import MLPEncoder
from gcn import GCN

class Model(nn.Module):
    def __init__(self, num_classes=12, backbone='resnet50'):
        super(Model, self).__init__()
        # 如果选择resnet50 作为 backbone，则为下面的
        self.mefarg = MEFARG(num_classes=num_classes, backbone=backbone)

        if backbone == 'resnet50':
            self.classifier1 = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1024),
            )
            self.classifier2 = nn.Sequential(
                nn.Linear(in_features=2048, out_features=128),
            )
            self.g_dim = 1024
        if backbone == 'swin_transformer_small':
            self.classifier1 = nn.Sequential(
                nn.Linear(in_features=768, out_features=768),
            )
            self.classifier2 = nn.Sequential(
                nn.Linear(in_features=768, out_features=128),
            )
            self.g_dim = 768
        if backbone == 'swin_transformer_base':
            self.classifier1 = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1024),
            )
            self.classifier2 = nn.Sequential(
                nn.Linear(in_features=1024, out_features=128),
            )
            self.g_dim = 1024
        self.relu = nn.ReLU()
        self.gcn = GCN(num_class=num_classes,)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_classes * 49 * 128, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        # 初始化 classifier1 和 classifier 中的线性层
        for m in self.classifier1.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        for m in self.classifier2.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mefarg(x)
        x1 = self.classifier2(x)
        x1 = self.relu(x1)
        x = self.classifier1(x)
        x = self.relu(x)
        x = self.gcn(x)
        x = x + x1
        x = torch.flatten(x, 1)
        result = self.classifier(x)
        m = nn.Sigmoid()
        result = m(result)
        return result




