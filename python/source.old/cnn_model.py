from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling1D,
    ZeroPadding1D,
    Conv1D,
    ZeroPadding2D,
    MaxPooling2D,
)
import tensorflow.keras.optimizers as opt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import f1_score

import numpy as np
import pickle
import sys
import random


def load_data(feature_path, label_path):
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)

    with open(label_path, "rb") as f:
        label = pickle.load(f)

    return feature, label


def vgg_19(input_len):
    model = Sequential()
    model.add(
        ZeroPadding2D(padding=(1, 1), input_shape=(input_len[1], input_len[2], 1))
    )
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 1), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 1), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 1), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 1), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 1), strides=(2, 2)))

    # model.add(ZeroPadding2D(padding=(1, 1)))
    # model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    # model.add(ZeroPadding2D(padding=(1, 1)))
    # model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    # model.add(ZeroPadding2D(padding=(1, 1)))
    # model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    # model.add(ZeroPadding2D(padding=(1, 1)))
    # model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D((1, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    if RESEARCH_QUESTION == "q1":
        model.add(Dense(3, activation="softmax"))
    if RESEARCH_QUESTION == "q3":
        model.add(Dense(2, activation="softmax"))

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
        (train_feature.shape[0], train_feature.shape[1], train_feature.shape[2], 1)
    )

    test_feature, test_label = load_data(TEST_FEATURE_PATH, TEST_LABEL_PATH)
    test_feature_ = test_feature.reshape(
        (test_feature.shape[0], test_feature.shape[1], train_feature.shape[2], 1)
    )

    train_len = train_label.shape[0]
    test_len = test_label.shape[0]

    all_label = np.concatenate((train_label, test_label))

    train_onehot, test_labels = encode_onehot(all_label, train_len)

    # Test pretrained model
    model = vgg_19(train_feature_.shape)

    # Optimizers
    sgd = opt.SGD(lr=0.01, momentum=0.5, nesterov=False)
    adam = opt.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    rms_prop = opt.RMSprop(lr=0.01, rho=0.9)
    adagrad = opt.Adagrad(lr=0.01)
    adadelta = opt.Adadelta(lr=1.0, rho=0.95)
    adamax = opt.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)
    nadam = opt.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=sgd, loss="categorical_crossentropy")

    print(model.summary())

    model.fit(
        train_feature_,
        train_onehot.toarray(),
        batch_size=64,
        # batch_size=1775,
        epochs=30,
    )

    prediction = model.predict(test_feature_)
    predicted_label = np.argmax(prediction, axis=1).reshape((-1, 1))

    err_array = np.subtract(predicted_label, test_labels)
    err_idx = np.where(err_array != 0)[1]

    err = round(((len(err_idx) / test_len) * 100), 2)
    acc = 100 - err

    f1 = f1_score(test_label, predicted_label, average="macro")
    f1 = round((f1 * 100), 2)

    print("CNN model's accuracy is ", acc, "%")
    print("CNN model's F1 score is ", f1, "%")
