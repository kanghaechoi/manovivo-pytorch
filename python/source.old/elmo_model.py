import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer
import numpy as np
import pickle
import sys

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module(
            "https://tfhub.dev/google/elmo/2",
            trainable=self.trainable,
            name="{}_module".format(self.name),
        )

        self.trainable_weights += K.tf.trainable_variables(
            scope="^{}_module/.*".format(self.name)
        )
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            K.squeeze(K.cast(x, tf.string), axis=1),
            as_dict=True,
            signature="default",
        )["default"]
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, "--PAD--")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


def load_data(feature_path, label_path):
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)

    with open(label_path, "rb") as f:
        label = pickle.load(f)

    return feature, label


def elmo():
    input_text = layers.Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = layers.Dense(256, activation="relu")(embedding)
    pred = layers.Dense(1, activation="sigmoid")(dense)

    model = Model(inputs=[input_text], outputs=pred)

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()

    return model


def encode_onehot(label, train_len):
    label = np.squeeze(label.reshape((1, -1)))

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(label)
    encoded_label = encoded_label.reshape((-1, 1))

    train_label = encoded_label[:train_len, 0].reshape((-1, 1))
    test_label = encoded_label[train_len:, 0].reshape((-1, 1))

    onehot_encoder = OneHotEncoder()
    train_onehot = onehot_encoder.fit_transform(train_label)
    # print(encode_label.classes_)
    # print(encode_label.transform(label))

    return train_onehot, test_label


def encode_label(label):
    onehot_encoder = OneHotEncoder()
    onehot_label = onehot_encoder.fit_transform(label)
    # print(encode_label.classes_)
    # print(encode_label.transform(label))

    return onehot_label


if __name__ == "__main__":
    # argument = sys.argv
    # del argument[0]

    RESEARCH_QUESTION = argument[0]
    IS_DEBUG = argument[1]

    # RESEARCH_QUESTION = 'q1'
    # IS_DEBUG = 'unix'

    if IS_DEBUG == "n":
        TRAIN_FEATURE_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/train_feature_seq.pickle"
        )
        TEST_FEATURE_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/test_feature_seq.pickle"
        )

        TRAIN_LABEL_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/train_label_seq.pickle"
        )
        TEST_LABEL_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/test_label_seq.pickle"
        )

    if IS_DEBUG == "y":
        TRAIN_FEATURE_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/train_feature_seq.pickle"
        )
        TEST_FEATURE_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/test_feature_seq.pickle"
        )

        TRAIN_LABEL_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/train_label_seq.pickle"
        )
        TEST_LABEL_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/test_label_seq.pickle"
        )

    train_feature, train_label = load_data(TRAIN_FEATURE_PATH, TRAIN_LABEL_PATH)
    train_feature_ = train_feature.reshape(
        (train_feature.shape[2], train_feature.shape[0], train_feature.shape[1])
    )

    test_feature, test_label = load_data(TEST_FEATURE_PATH, TEST_LABEL_PATH)
    test_feature_ = test_feature.reshape(
        (test_feature.shape[2], test_feature.shape[0], train_feature.shape[1])
    )

    train_len = train_label.shape[0]
    test_len = test_label.shape[0]

    all_label = np.concatenate((train_label, test_label))

    train_onehot, test_labels = encode_onehot(all_label, train_len)

    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    model = elmo()

    model.fit(train_feature_, train_onehot.toarray(), epochs=10, batch_size=32)

    predicted_label = np.argmax(model.predict(test_feature_), axis=1).reshape(
        (-1, 1)
    )

    err_array = np.subtract(predicted_label, test_labels)
    err_idx = np.where(err_array != 0)[1]

    err = round(((len(err_idx) / test_len) * 100), 2)
    acc = 100 - err

    print("RNN model's accuracy is ", acc, "%")

