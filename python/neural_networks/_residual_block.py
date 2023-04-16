import torch.nn as nn
import torch.nn.functional as F


class _BottleneckBase(nn.Module):
    def __init__(self, _in_channels: int, _stride: int = 1) -> None:
        super().__init__()

        self.convolution_layer_1 = nn.Conv2d(
            in_channels=_in_channels,
            out_channels=_in_channels,
            kernel_size=(1, 1),
            stride=1,
            # padding="same",
        )
        self.convolution_layer_2 = nn.Conv2d(
            in_channels=_in_channels,
            out_channels=_in_channels,
            kernel_size=(3, 3),
            stride=_stride,
            # padding="same",
        )
        self.convolution_layer_3 = nn.Conv2d(
            in_channels=_in_channels,
            out_channels=_in_channels * 4,
            kernel_size=(1, 1),
            stride=1,
            # padding="same",
        )

        self.batch_normalization_1 = nn.BatchNorm1d(num_features=_in_channels)
        self.relu_1 = nn.ReLU()

        self.batch_normalization_2 = nn.BatchNorm1d(num_features=_in_channels)
        self.relu_2 = nn.ReLU()

        self.batch_normalization_3 = nn.BatchNorm1d(num_features=_in_channels * 4)
        self.relu_3 = nn.ReLU()

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=_in_channels,
                out_channels=_in_channels * 4,
                kernel_size=(1, 1),
                stride=_stride,
            ),
            nn.BatchNorm1d(num_features=_in_channels * 4),
        )


class BottleneckType1(_BottleneckBase):
    def __init__(self, _channels: int, _strides: int) -> None:
        super().__init__(_channels, _strides)

    def forward(self, input) -> nn.Module:
        residual = self.shortcut(input)

        x = self.convolution_layer_1(input)
        x = self.batch_normalization_1(x)
        x = self.relu_1(x)
        x = self.convolution_layer_2(x)
        x = self.batch_normalization_2(x)
        x = self.relu_2(x)
        x = self.convolution_layer_3(x)
        x = self.batch_normalization_3(x)

        output = residual.add_module(x)
        output = self.relu_3(output)

        return output


class BottleneckType2(_BottleneckBase):
    def __init__(self, _channels: int, _strides: int) -> None:
        super().__init__(_channels, _strides)
        self.max_pooling = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2,
            padding=0,
        )

    def forward(self, input) -> nn.Module:
        residual = self.shortcut(input)

        x = self.max_pooling(input)
        x = self.convolution_layer_1(input)
        x = self.batch_normalization_1(x)
        x = self.relu_1(x)
        x = self.convolution_layer_2(x)
        x = self.batch_normalization_2(x)
        x = self.relu_2(x)
        x = self.convolution_layer_3(x)
        x = self.batch_normalization_3(x)

        output = residual.add_module(x)
        output = self.relu_3(output)

        return output
