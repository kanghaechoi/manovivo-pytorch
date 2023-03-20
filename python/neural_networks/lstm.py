import tensorflow as tf


class LSTM(tf.keras.Model):
    def __init__(self, _units: int):
        super(LSTM, self).__init__()

        self.forward_layer = tf.keras.layers.LSTM(units=_units, return_sequences=True)

        self.dense_layer = tf.keras.layers.Dense(units=_units)
        self.activation_layer = tf.keras.layers.Activation("softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.forward_layer(inputs)
        x = self.dense_layer(x)
        output = self.activation_layer(x)

        return output


class BidirectionalLSTM(LSTM):
    def __init__(self, _units: int):
        super(BidirectionalLSTM, self).__init__()

        self.bidirectional_layer = tf.keras.layers.Bidirectional(self.forward_layer)

        self.dense_layer = tf.keras.layers.Dense(units=_units)
        self.activation_layer = tf.keras.layers.Activation("softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.bidirectional_layer(inputs)
        x = self.dense_layer(x)
        output = self.activation_layer(x)

        return output
