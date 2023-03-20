import torch.nn as nn


class _BottleneckBase(nn.Module):
    def __init__(self, _channels: int, _strides: int = 1):
        super().__init__()

        self.convolution_layer_type1 = nn.Conv2d(
            filters=_channels,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
        )
        self.convolution_layer_type2 = nn.Conv2d(
            filters=_channels,
            kernel_size=(3, 3),
            strides=_strides,
            padding="same",
        )
        self.convolution_layer_type3 = nn.Conv2d(
            filters=_channels * 4,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
        )

        self.shortcut = nn.Sequential()
        self.shortcut.append(
            nn.Conv2d(
                filters=_channels * 4,
                kernel_size=(1, 1),
                strides=_strides,
            )
        )
        self.shortcut.append(nn.BatchNorm1d())


class BottleneckType1(_BottleneckBase):
    def __init__(self, _channels: int, _strides: int):
        super().__init__(_channels, _strides)

    def call(self, input, **kwargs):
        residual = self.shortcut(input)

        x = self.convolution_layer_type1(input)
        x = nn.BatchNorm1d(x)
        x = nn.ReLU(x)
        x = self.convolution_layer_type2(x)
        x = nn.BatchNorm1d(x)
        x = nn.ReLU(x)
        x = self.convolution_layer_type3(x)
        x = nn.BatchNorm1d(x)

        output = residual.add_module(x)
        output = nn.ReLU(output)

        return output


class BottleneckType2(_BottleneckBase):
    def __init__(self, _channels: int, _strides: int):
        super().__init__(_channels, _strides)
        self.max_pooling = nn.MaxPool2d(
            pool_size=(3, 3),
            strides=2,
            padding="same",
        )

    def call(self, input, **kwargs):
        residual = self.shortcut(input)

        x = self.max_pooling(input)
        x = self.convolution_layer_type1(input)
        x = nn.BatchNorm1d(x)
        x = nn.ReLU(x)
        x = self.convolution_layer_type2(x)
        x = nn.BatchNorm1d(x)
        x = nn.ReLU(x)
        x = self.convolution_layer_type3(x)
        x = nn.BatchNorm1d(x)

        output = residual.add_module(x)
        output = nn.ReLU(output)

        return output
