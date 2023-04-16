import torch
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
    def __init__(self, _block_parameters: list[int], _dim: int) -> None:
        super().__init__()

        self.convolution_layer_1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=2,
            # padding="same",
        )

        self.batch_normalization_1 = nn.BatchNorm1d(64)
        self.relu_1 = nn.ReLU()

        self.convolution_layer_2 = residual_convolution_layer_type1(
            in_channels=64,
            blocks=_block_parameters[0],
        )
        self.convolution_layer_3 = residual_convolution_layer_type2(
            in_channels=128,
            blocks=_block_parameters[1],
        )
        self.convolution_layer_4 = residual_convolution_layer_type2(
            in_channels=256,
            blocks=_block_parameters[2],
        )
        self.convolution_layer_5 = residual_convolution_layer_type2(
            in_channels=512,
            blocks=_block_parameters[3],
        )

        self.average_pooling = nn.AvgPool2d(kernel_size=(3, 3))

        self.fully_connected_layer = nn.Softmax(dim=_dim)

    def forward(self, input: torch.Tensor) -> nn.Module:
        x = self.convolution_layer_1(input)
        x = self.batch_normalization_1(x)
        x = self.relu_1(x)

        x = self.convolution_layer_2(x)
        x = self.convolution_layer_3(x)
        x = self.convolution_layer_4(x)
        x = self.convolution_layer_5(x)

        x = self.average_pooling(x)

        output = self.fully_connected_layer(x)

        return output
