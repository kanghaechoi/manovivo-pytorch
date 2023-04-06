from typing import List

import torch.nn as nn
import torch.nn.functional as F

from neural_networks._residual_block import (
    BottleneckType1,
    BottleneckType2,
)


def residual_convolution_layer_type1(in_channels: int, blocks: int, stride: int = 1):
    residual_convolution_layer = nn.Sequential()
    residual_convolution_layer.append(BottleneckType2(in_channels, stride))

    for _ in range(1, blocks):
        residual_convolution_layer.append(BottleneckType1(in_channels, 1))

    return residual_convolution_layer


def residual_convolution_layer_type2(in_channels: int, blocks: int, stride: int = 1):
    residual_convolution_layer = nn.Sequential()
    residual_convolution_layer.append(BottleneckType1(in_channels, stride))

    for _ in range(1, blocks):
        residual_convolution_layer.append(BottleneckType1(in_channels, 1))

    return residual_convolution_layer


class ResNet(nn.Module):
    def __init__(self, _block_parameters: List[int], _dim: int) -> None:
        super().__init__()

        self.convolution_layer_type_1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=2,
            padding="same",
        )

        self.convolution_layer_type2 = residual_convolution_layer_type1(
            in_channels=64,
            blocks=_block_parameters[0],
        )
        self.convolution_layer_type3 = residual_convolution_layer_type2(
            in_channels=128,
            blocks=_block_parameters[1],
        )
        self.convolution_layer_type4 = residual_convolution_layer_type2(
            in_channels=256,
            blocks=_block_parameters[2],
        )
        self.convolution_layer_type5 = residual_convolution_layer_type2(
            in_channels=512,
            blocks=_block_parameters[3],
        )

        self.fully_connected_layer = nn.Softmax(dim=_dim)

    def forward(self, input) -> nn.Module:
        x = self.convolution_layer_type_1(input)
        x = F.batch_norm(x)
        x = F.relu(x)

        x = self.convolution_layer_type2(x)
        x = self.convolution_layer_type3(x)
        x = self.convolution_layer_type4(x)
        x = self.convolution_layer_type5(x)

        x = F.adaptive_avg_pool2d(x)

        output = self.fully_connected_layer(x)

        return output
