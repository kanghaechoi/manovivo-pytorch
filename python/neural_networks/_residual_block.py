import torch.nn as nn
import torch.nn.functional as F


class _BottleneckBase(nn.Module):
    def __init__(self, _in_channels: int, _stride: int = 1) -> None:
        super().__init__()

        self.convolution_layer_type1 = nn.Conv2d(
            in_channels=_in_channels,
            out_channels=_in_channels,
            kernel_size=(1, 1),
            stride=1,
            padding="same",
        )
        self.convolution_layer_type2 = nn.Conv2d(
            in_channels=_in_channels,
            out_channels=_in_channels,
            kernel_size=(3, 3),
            stride=_stride,
            padding="same",
        )
        self.convolution_layer_type3 = nn.Conv2d(
            in_channels=_in_channels,
            out_channels=_in_channels * 4,
            kernel_size=(1, 1),
            stride=1,
            padding="same",
        )

        self.shortcut = nn.Sequential()
        self.shortcut.append(
            nn.Conv2d(
                in_channels=_in_channels,
                out_channels=_in_channels * 4,
                kernel_size=(1, 1),
                stride=_stride,
            )
        )
        self.shortcut.append(nn.BatchNorm1d())


class BottleneckType1(_BottleneckBase):
    def __init__(self, _channels: int, _strides: int) -> None:
        super().__init__(_channels, _strides)

    def forward(self, input) -> nn.Module:
        residual = self.shortcut(input)

        x = self.convolution_layer_type1(input)
        x = F.batch_norm(x)
        x = F.relu(x)
        x = self.convolution_layer_type2(x)
        x = F.batch_norm(x)
        x = F.relu(x)
        x = self.convolution_layer_type3(x)
        x = F.batch_norm(x)

        output = residual.add_module(x)
        output = F.relu(output)

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
        x = self.convolution_layer_type1(input)
        x = F.batch_norm(x)
        x = F.relu(x)
        x = self.convolution_layer_type2(x)
        x = F.batch_norm(x)
        x = F.relu(x)
        x = self.convolution_layer_type3(x)
        x = F.batch_norm(x)

        output = residual.add_module(x)
        output = F.relu(output)

        return output
