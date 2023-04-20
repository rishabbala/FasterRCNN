import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict

from typing import List


class BBPredNet(nn.Module):
    
    def __init__(self, num_anchor_types: int):
        super().__init__()
        
        self.num_anchor_types = num_anchor_types
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.c_conv = nn.Conv2d(in_channels=256, out_channels=2 * self.num_anchor_types, kernel_size=1, padding=0)
        self.yx_conv = nn.Conv2d(in_channels=256, out_channels=2 * self.num_anchor_types, kernel_size=1, padding=0)
        self.hw_conv = nn.Conv2d(in_channels=256, out_channels=2 * self.num_anchor_types, kernel_size=1, padding=0)

        self.relu = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        
        fmap_class = self.c_conv(x)
        fmap_class = fmap_class.view(fmap_class.shape[0], 2, self.num_anchor_types, fmap_class.shape[2], fmap_class.shape[3])
        
        fmap_box_yx = self.yx_conv(x)
        fmap_box_yx = self.relu(fmap_box_yx.view(fmap_box_yx.shape[0], 2, self.num_anchor_types, fmap_box_yx.shape[2], fmap_box_yx.shape[3]))

        fmap_box_hw = self.hw_conv(x)
        fmap_box_hw = self.relu(fmap_box_hw.view(fmap_box_hw.shape[0], 2, self.num_anchor_types, fmap_box_hw.shape[2], fmap_box_hw.shape[3])) + 1e-3
        
        return fmap_class, fmap_box_yx, fmap_box_hw

    
class ROIPredNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=20)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x).view(x.shape[0], -1)
        x = self.linear_layers(x)
        
        return x