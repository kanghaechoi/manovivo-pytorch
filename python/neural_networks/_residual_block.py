import tensorflow as tf


class _BottleneckBase(tf.keras.Model):
    def __init__(self, _channels: int, _strides: int = 1):
        super().__init__()

        self.convolution_layer_type1 = tf.keras.layers.Conv2D(
            filters=_channels,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
        )
        self.convolution_layer_type2 = tf.keras.layers.Conv2D(
            filters=_channels,
            kernel_size=(3, 3),
            strides=_strides,
            padding="same",
        )
        self.convolution_layer_type3 = tf.keras.layers.Conv2D(
            filters=_channels * 4,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
        )

        self.batch_normalization_type1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization_type2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization_type3 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        self.shortcut.add(
            tf.keras.layers.Conv2D(
                filters=_channels * 4,
                kernel_size=(1, 1),
                strides=_strides,
            )
        )
        self.shortcut.add(tf.keras.layers.BatchNormalization())


class BottleneckType1(_BottleneckBase):
    def __init__(self, _channels: int, _strides: int):
        super().__init__(_channels, _strides)

    def call(self, input, **kwargs):
        residual = self.shortcut(input)

        x = self.convolution_layer_type1(input)
        x = self.batch_normalization_type1(x)
        x = tf.nn.relu(x)
        x = self.convolution_layer_type2(x)
        x = self.batch_normalization_type2(x)
        x = tf.nn.relu(x)
        x = self.convolution_layer_type3(x)
        x = self.batch_normalization_type3(x)

        output = tf.keras.layers.add([residual, x])
        output = tf.nn.relu(output)

        return output


class BottleneckType2(_BottleneckBase):
    def __init__(self, _channels: int, _strides: int):
        super().__init__(_channels, _strides)
        self.max_pooling = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
            padding="same",
        )

    def call(self, input, **kwargs):
        residual = self.shortcut(input)

        x = self.max_pooling(input)
        x = self.convolution_layer_type1(input)
        x = self.batch_normalization_type1(x)
        x = tf.nn.relu(x)
        x = self.convolution_layer_type2(x)
        x = self.batch_normalization_type2(x)
        x = tf.nn.relu(x)
        x = self.convolution_layer_type3(x)
        x = self.batch_normalization_type3(x)

        output = tf.keras.layers.add([residual, x])
        output = tf.nn.relu(output)

        return output
