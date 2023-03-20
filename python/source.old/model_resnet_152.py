from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.regularizers import l2
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix

import numpy as np
import pickle
import sys
import time
import random


def load_data(feature_path, label_path):
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)

    with open(label_path, "rb") as f:
        label = pickle.load(f)

    return feature, label


def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    return x


def conv2_layer(x):
    x = MaxPooling2D((3, 3), 2)(x)

    shortcut = x

    for i in range(3):
        if i == 0:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

    return x


def conv3_layer(x):
    shortcut = x

    for i in range(8):
        if i == 0:
            x = Conv2D(128, (1, 1), strides=(2, 2), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding="valid")(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding="valid")(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

    return x


def conv4_layer(x):
    shortcut = x

    for i in range(36):
        if i == 0:
            x = Conv2D(256, (1, 1), strides=(2, 2), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding="valid")(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding="valid")(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

    return x


def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if i == 0:
            x = Conv2D(512, (1, 1), strides=(2, 2), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding="valid")(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding="valid")(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding="valid")(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation("relu")(x)

            shortcut = x

    return x


def resnet(input_len):
    if RESEARCH_QUESTION == "q1":
        num_class = 3
    if RESEARCH_QUESTION == "q2":
        num_class = 2
    if RESEARCH_QUESTION == "q3":
        num_class = 2

    input_tensor = Input(
        shape=(1, input_len[2], input_len[3]), dtype="float32", name="input"
    )

    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)

    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(num_class, activation="softmax")(x)

    model = Model(input_tensor, output_tensor)

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
    argument = sys.argv
    del argument[0]

    RESEARCH_QUESTION = argument[0]
    IS_DEBUG = argument[1]

    # RESEARCH_QUESTION = str('q1')
    # IS_DEBUG = 'y'

    if IS_DEBUG == "n":
        TRAIN_FEATURE_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/train_feature_seq.pickle"
        )
        TEST_FEATURE_PATH = "./pickle/" + RESEARCH_QUESTION + "/test_feature_seq.pickle"

        TRAIN_LABEL_PATH = "./pickle/" + RESEARCH_QUESTION + "/train_label_seq.pickle"
        TEST_LABEL_PATH = "./pickle/" + RESEARCH_QUESTION + "/test_label_seq.pickle"

    if IS_DEBUG == "y":
        TRAIN_FEATURE_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/train_feature_seq.pickle"
        )
        TEST_FEATURE_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/test_feature_seq.pickle"
        )

        TRAIN_LABEL_PATH = "../pickle/" + RESEARCH_QUESTION + "/train_label_seq.pickle"
        TEST_LABEL_PATH = "../pickle/" + RESEARCH_QUESTION + "/test_label_seq.pickle"

    train_feature, train_label = load_data(TRAIN_FEATURE_PATH, TRAIN_LABEL_PATH)

    # all_idx = np.linspace(0, (train_feature.shape[2] - 1), num=train_feature.shape[2], dtype=int)
    # random.shuffle(all_idx)
    #
    # train_feature = train_feature[:, :, all_idx]
    # train_label = train_label[all_idx, :]

    train_feature_ = train_feature.reshape(
        (train_feature.shape[0], 1, train_feature.shape[1], train_feature.shape[2])
    )

    test_feature, test_label = load_data(TEST_FEATURE_PATH, TEST_LABEL_PATH)
    test_feature_ = test_feature.reshape(
        (test_feature.shape[0], 1, test_feature.shape[1], train_feature.shape[2])
    )

    train_len = train_label.shape[0]
    test_len = test_label.shape[0]

    all_label = np.concatenate((train_label, test_label))

    train_onehot, test_labels = encode_onehot(all_label, train_len)

    # Test pretrained model
    model = resnet(train_feature_.shape)

    # Optimizers
    sgd = opt.SGD(lr=0.01, momentum=0.5, nesterov=False)
    adam = opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    rms_prop = opt.RMSprop(lr=0.01, rho=0.9)
    adagrad = opt.Adagrad(lr=0.01)
    adadelta = opt.Adadelta(lr=1.0, rho=0.95)
    adamax = opt.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
    nadam = opt.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=adam, loss="categorical_crossentropy")

    # print(model.summary())
    print("ResNet-152")

    train_start = time.time()
    model.fit(
        train_feature_,
        train_onehot.toarray(),
        batch_size=128,
        # batch_size=1775,
        epochs=500,
    )
    train_end = time.time()

    pred_start = time.time()
    prediction = model.predict(test_feature_)
    pred_end = time.time()

    predicted_label = np.argmax(prediction, axis=1).reshape((-1, 1))

    train_time = train_end - train_start
    pred_time = (pred_end - pred_start) / predicted_label.shape[0]
    print(train_time)
    print(pred_time)

    err_array = np.subtract(predicted_label, test_labels)
    err_idx = np.where(err_array != 0)[1]

    err = round(((len(err_idx) / test_len) * 100), 2)
    acc = 100 - err

    # f1 = f1_score(test_labels, predicted_label, average="binary")
    # f1 = round((f1 * 100), 2)

    conf_mat = confusion_matrix(test_labels, predicted_label)

    print(conf_mat)

    if RESEARCH_QUESTION == "q1":
        (
            conf_mat_11,
            conf_mat_12,
            conf_mat_13,
            conf_mat_21,
            conf_mat_22,
            conf_mat_23,
            conf_mat_31,
            conf_mat_32,
            conf_mat_33,
        ) = conf_mat.ravel()

        len_20 = np.where(test_labels == 0)[0].size
        len_50 = np.where(test_labels == 1)[0].size
        len_70 = np.where(test_labels == 2)[0].size

        accu = (conf_mat_11 + conf_mat_22 + conf_mat_33) / np.sum(conf_mat.ravel())
        recall = (
            (len_20 * (conf_mat_11 / np.sum(conf_mat[:, 0])))
            + (len_50 * (conf_mat_22 / np.sum(conf_mat[:, 1])))
            + (len_70 * (conf_mat_33 / np.sum(conf_mat[:, 2])))
        ) / (len_20 + len_50 + len_70)
        precision = (
            (len_20 * (conf_mat_11 / np.sum(conf_mat[0, :])))
            + (len_50 * (conf_mat_22 / np.sum(conf_mat[1, :])))
            + (len_70 * (conf_mat_33 / np.sum(conf_mat[2, :])))
        ) / (len_20 + len_50 + len_70)
        f1_s = (2 * precision * recall) / (precision + recall)
        specificity = (
            (len_20 * (np.sum(conf_mat[1:3, 1:3]) / np.sum(conf_mat[:, 1:3])))
            + (
                len_50
                * (
                    (conf_mat_11 + conf_mat_13 + conf_mat_31 + conf_mat_33)
                    / (np.sum(conf_mat[:, 0]) + np.sum(conf_mat[:, 2]))
                )
            )
            + (len_70 * (np.sum(conf_mat[0:2, 0:2]) / np.sum(conf_mat[:, 0:2])))
        ) / (len_20 + len_50 + len_70)

        far = (
            (len_20 * (np.sum(conf_mat[0, 1:3]) / np.sum(conf_mat[:, 1:3])))
            + (
                len_50
                * (
                    (conf_mat_21 + conf_mat_23)
                    / (np.sum(conf_mat[:, 0]) + np.sum(conf_mat[:, 2]))
                )
            )
            + (len_70 * (np.sum(conf_mat[2, 0:2]) / np.sum(conf_mat[:, 0:2])))
        ) / (len_20 + len_50 + len_70)
        frr = (
            (len_20 * (np.sum(conf_mat[1:3, 0]) / np.sum(conf_mat[:, 0])))
            + (len_50 * ((conf_mat_12 + conf_mat_32) / np.sum(conf_mat[:, 1])))
            + (len_70 * (np.sum(conf_mat[0:2, 2]) / np.sum(conf_mat[:, 2])))
        ) / (len_20 + len_50 + len_70)

    if RESEARCH_QUESTION == "q2":
        tn, fp, fn, tp = conf_mat.ravel()

        accu = (tn + tp) / np.sum(conf_mat.ravel())
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_s = (2 * precision * recall) / (precision + recall)
        specificity = tn / (fp + tn)
        far = fp / (fp + tn)
        frr = fn / (fn + tp)

    if RESEARCH_QUESTION == "q3":
        tn, fp, fn, tp = conf_mat.ravel()

        accu = (tn + tp) / np.sum(conf_mat.ravel())
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_s = (2 * precision * recall) / (precision + recall)
        specificity = tn / (fp + tn)
        far = fp / (fp + tn)
        frr = fn / (fn + tp)

    print("CNN model's accuracy is ", acc, "%")
    # print("CNN model's F1 score is ", f1, "%")
    print(accu)
    # print(specificity)
    print(recall)
    print(precision)
    print(f1_s)
    print(far)
    print(frr)