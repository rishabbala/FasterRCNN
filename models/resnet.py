import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict

from typing import List


class MakeResnet(nn.Module):
    """ 
    Make the resnet module
    """

    def __init__(self, num_layers: List[int]=None):

        super().__init__()
        self.resnet = ResnetBasic(num_layers=num_layers)
    
    def forward(self, x: Tensor) -> Tensor:
        
        z = self.resnet(x)

        return z



class ResnetBasic(nn.Module):


    def __init__(self, num_layers: List[int]):
        
        super().__init__()

        self.encoder = nn.Sequential(OrderedDict([
                                    ('conv', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)),
                                    ('norm', nn.BatchNorm2d(num_features=64)),
                                    ('relu', nn.ReLU(inplace=True))
                                    ]))

        self.layer0 = self._make_layer(num_layers[0], in_channels=64, out_channels=64, padding=1, stride=1)
        self.layer1 = self._make_layer(num_layers[1], in_channels=64, out_channels=128, padding=1, stride=2)
        self.layer2 = self._make_layer(num_layers[2], in_channels=128, out_channels=256, padding=1, stride=2)
        self.layer3 = self._make_layer(num_layers[3], in_channels=256, out_channels=512, padding=1, stride=2)


    def _make_layer(self, n_layer: int, in_channels: int, out_channels: int, padding: int, stride: int) -> nn.Sequential:

        layer = []
        stride_layer = []
        channel_layer = []

        for _ in range(n_layer-1):
            stride_layer.append(1)
            channel_layer.append(out_channels)

        stride_layer.insert(0, stride)
        channel_layer.insert(0, in_channels)

        for i in range(n_layer):
            layer.append(BasicBlock(channel_layer[i], out_channels, padding=padding, stride=stride_layer[i]))

        return nn.Sequential(*layer)


    def forward(self, x: Tensor, train: bool=True) -> Tensor:

        x = self.encoder(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class BasicBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, padding: int, stride: int):
        super().__init__()

        self.downsample = None
        self.relu = nn.ReLU()

        self.block = nn.Sequential(OrderedDict([
                                    ('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, stride=stride)),
                                    ('norm1', nn.BatchNorm2d(num_features=out_channels)),
                                    ('relu1', nn.ReLU(inplace=True)),
                                    ('conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)),
                                    ('norm2', nn.BatchNorm2d(num_features=out_channels))
                                    ]))

        if stride != 1:
            self.downsample = nn.Sequential(OrderedDict([
                                            ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)),
                                            ('norm', nn.BatchNorm2d(num_features=out_channels))
                                            ]))

    
    def forward(self, x: Tensor) -> Tensor:

        y = self.block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        y = x + y
        y = self.relu(y)

        return y