from typing import List

import tensorflow as tf

from neural_networks._residual_block import (
    BottleneckType1,
    BottleneckType2,
)


def residual_convolution_layer_type1(channels: int, blocks: int, strides: int = 1):
    residual_convolution_layer = tf.keras.Sequential()
    residual_convolution_layer.add(BottleneckType2(channels, strides))

    for _ in range(1, blocks):
        residual_convolution_layer.add(BottleneckType1(channels, 1))

    return residual_convolution_layer


def residual_convolution_layer_type2(channels: int, blocks: int, strides: int = 1):
    residual_convolution_layer = tf.keras.Sequential()
    residual_convolution_layer.add(BottleneckType1(channels, strides))

    for _ in range(1, blocks):
        residual_convolution_layer.add(BottleneckType1(channels, 1))

    return residual_convolution_layer


class ResNet(tf.keras.Model):
    def __init__(self, _block_parameters: List[int], _units: int):
        super(ResNet, self).__init__()

        self.convolution_layer_type_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
        )
        self.batch_normalization = tf.keras.layers.BatchNormalization()

        self.convolution_layer_type2 = residual_convolution_layer_type1(
            channels=64,
            blocks=_block_parameters[0],
        )
        self.convolution_layer_type3 = residual_convolution_layer_type2(
            channels=128,
            blocks=_block_parameters[1],
        )
        self.convolution_layer_type4 = residual_convolution_layer_type2(
            channels=256,
            blocks=_block_parameters[2],
        )
        self.convolution_layer_type5 = residual_convolution_layer_type2(
            channels=512,
            blocks=_block_parameters[3],
        )

        self.average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.fully_connected_layer = tf.keras.layers.Dense(
            units=_units,
            activation=tf.keras.activations.softmax,
        )

    def call(self, inputs, training=None, mask=None) -> any:
        x = self.convolution_layer_type_1(inputs)
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)

        x = self.convolution_layer_type2(x)
        x = self.convolution_layer_type3(x)
        x = self.convolution_layer_type4(x)
        x = self.convolution_layer_type5(x)

        x = self.average_pooling(x)

        output = self.fully_connected_layer(x)

        return output
