import torch
import torch.nn as nn

from neural_networks.resnet.block import (
    Base,
    Bottleneck,
)


class ResNet(nn.Module):
    def __init__(self, _block_parameters: list[int], _number_of_classes: int) -> None:
        super().__init__()

        _expansion = 4

        self.in_channels = 64

        self.convolution_layer_1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.batch_normalization_layer_1 = nn.BatchNorm2d(num_features=64)

        self.max_pooling_layer = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.resnet_layer_2 = self.build_block(
            layer=Bottleneck,
            out_channels=64,
            stride=1,
            number_of_blocks=_block_parameters[0],
        )
        self.resnet_layer_3 = self.build_block(
            layer=Bottleneck,
            out_channels=128,
            stride=2,
            number_of_blocks=_block_parameters[1],
        )
        self.resnet_layer_4 = self.build_block(
            layer=Bottleneck,
            out_channels=256,
            stride=2,
            number_of_blocks=_block_parameters[2],
        )
        self.resnet_layer_5 = self.build_block(
            layer=Bottleneck,
            out_channels=512,
            stride=2,
            number_of_blocks=_block_parameters[3],
        )

        self.average_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.relu = nn.ReLU(inplace=True)

        self.fully_connected_layer = nn.Linear(512 * _expansion, _number_of_classes)

        self.softmax_layer = nn.Softmax(dim=0)

    def build_block(
        self,
        layer: Base | Bottleneck,
        out_channels: int,
        stride: int,
        number_of_blocks: int,
    ):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * layer.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * layer.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * layer.expansion),
            )

        layers = []
        layers.append(layer(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * layer.expansion

        for _ in range(1, number_of_blocks):
            layers.append(layer(self.in_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # resnet_layer_1
        output = self.convolution_layer_1(input)
        output = self.batch_normalization_layer_1(output)
        output = self.relu(output)
        output = self.max_pooling_layer(output)

        output = self.resnet_layer_2(output)
        output = self.resnet_layer_3(output)
        output = self.resnet_layer_4(output)
        output = self.resnet_layer_5(output)

        output = self.average_pooling(output)
        output = torch.flatten(output, 1)

        output = self.fully_connected_layer(output)
        output = self.softmax_layer(output)

        return output
