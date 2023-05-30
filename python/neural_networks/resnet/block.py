from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _convolution_layer_1x1(
    _in_channels: int,
    _out_channels: int,
    _stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=_in_channels,
        out_channels=_out_channels,
        kernel_size=1,
        stride=_stride,
        bias=False,
    )


def _convolution_layer_3x3(
    _in_channels: int,
    _out_channels: int,
    _stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=_in_channels,
        out_channels=_out_channels,
        kernel_size=3,
        stride=_stride,
        padding=1,
        bias=False,
    )


class Base(nn.Module):
    def __init__(self, _in_channels: int, _out_channels: int, _stride: int) -> None:
        super().__init__()

        self.convolution_layer_1 = _convolution_layer_3x3(
            _in_channels,
            _out_channels,
            _stride,
        )
        self.batch_normalization_layer_1 = nn.BatchNorm2d(num_features=_out_channels)

        self.convolution_layer_2 = _convolution_layer_3x3(
            _in_channels,
            _out_channels,
        )
        self.batch_normalization_layer_2 = nn.BatchNorm2d(num_features=_out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = input

        output = self.convolution_layer_1(input)
        output = self.batch_normalization_layer_1(output)
        output = self.relu(output)
        output = self.convolution_layer_2(output)
        output = self.batch_normalization_layer_2(output)
        output = self.relu(output)

        output += residual
        output = self.relu(output)

        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        _in_channels: int,
        _out_channels: int,
        _stride: int,
        _downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.convolution_layer_1 = _convolution_layer_1x1(
            _in_channels,
            _out_channels,
        )
        self.batch_normalization_layer_1 = nn.BatchNorm2d(num_features=_out_channels)

        self.convolution_layer_2 = _convolution_layer_3x3(
            _out_channels,
            _out_channels,
            _stride,
        )
        self.batch_normalization_layer_2 = nn.BatchNorm2d(num_features=_out_channels)

        self.convolution_layer_3 = _convolution_layer_1x1(
            _out_channels,
            _out_channels * self.expansion,
        )
        self.batch_normalization_layer_3 = nn.BatchNorm2d(
            num_features=_out_channels * self.expansion,
        )

        self.relu = nn.ReLU(inplace=True)

        self.stride = _stride
        self.downsample = _downsample

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        identity = input

        output = self.convolution_layer_1(input)
        output = self.batch_normalization_layer_1(output)
        output = self.relu(output)
        output = self.convolution_layer_2(output)
        output = self.batch_normalization_layer_2(output)
        output = self.relu(output)
        output = self.convolution_layer_3(output)
        output = self.batch_normalization_layer_3(output)

        if self.downsample is not None:
            identity = self.downsample(input)

        output += identity
        output = self.relu(output)

        return output
