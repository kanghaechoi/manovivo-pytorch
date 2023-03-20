import pickle
import sys
import numpy as np

from sklearn import preprocessing


def read_pickle(path_1, path_2, path_3):
    with open(path_1, "rb") as f:
        data_1 = pickle.load(f)

    with open(path_2, "rb") as f:
        data_2 = pickle.load(f)

    with open(path_3, "rb") as f:
        data_3 = pickle.load(f)

    all_data = np.concatenate((data_1, data_2, data_3))

    return all_data


def zero_to_one(array):
    min_max_scaler = preprocessing.MinMaxScaler()
    array_transformed = min_max_scaler.fit_transform(array)

    return array_transformed


if __name__ == "__main__":
    argument = sys.argv
    del argument[0]

    RESEARCH_QUESTION = argument[0]
    IS_DEBUG = argument[1]

    # RESEARCH_QUESTION = 'q3'
    # IS_DEBUG = 'y'

    if IS_DEBUG == "n":
        if RESEARCH_QUESTION == "q1":
            FEATURE_20 = "./pickle/" + RESEARCH_QUESTION + "/20_feature_norm.pickle"
            FEATURE_50 = "./pickle/" + RESEARCH_QUESTION + "/50_feature_norm.pickle"
            FEATURE_70 = "./pickle/" + RESEARCH_QUESTION + "/70_feature_norm.pickle"

            LABEL_20 = "./pickle/" + RESEARCH_QUESTION + "/20_label_norm.pickle"
            LABEL_50 = "./pickle/" + RESEARCH_QUESTION + "/50_label_norm.pickle"
            LABEL_70 = "./pickle/" + RESEARCH_QUESTION + "/70_label_norm.pickle"

            FEATURE_ALL = "./pickle/" + RESEARCH_QUESTION + "/all_feature_norm.pickle"
            LABEL_ALL = "./pickle/" + RESEARCH_QUESTION + "/all_label_norm.pickle"

        if RESEARCH_QUESTION == "q2":
            FEATURE_0 = "./pickle/" + RESEARCH_QUESTION + "/0_feature_norm.pickle"
            FEATURE_1 = "./pickle/" + RESEARCH_QUESTION + "/1_feature_norm.pickle"

            LABEL_0 = "./pickle/" + RESEARCH_QUESTION + "/0_label_norm.pickle"
            LABEL_1 = "./pickle/" + RESEARCH_QUESTION + "/1_label_norm.pickle"

            FEATURE_ALL = "./pickle/" + RESEARCH_QUESTION + "/all_feature_norm.pickle"
            LABEL_ALL = "./pickle/" + RESEARCH_QUESTION + "/all_label_norm.pickle"

        if RESEARCH_QUESTION == "q3":
            FEATURE_0 = "./pickle/" + RESEARCH_QUESTION + "/0_feature_norm.pickle"
            FEATURE_1 = "./pickle/" + RESEARCH_QUESTION + "/1_feature_norm.pickle"

            LABEL_0 = "./pickle/" + RESEARCH_QUESTION + "/0_label_norm.pickle"
            LABEL_1 = "./pickle/" + RESEARCH_QUESTION + "/1_label_norm.pickle"

            FEATURE_ALL = "./pickle/" + RESEARCH_QUESTION + "/all_feature_norm.pickle"
            LABEL_ALL = "./pickle/" + RESEARCH_QUESTION + "/all_label_norm.pickle"

    if IS_DEBUG == "y":
        if RESEARCH_QUESTION == "q1":
            FEATURE_20 = "../pickle/" + RESEARCH_QUESTION + "/20_feature_norm.pickle"
            FEATURE_50 = "../pickle/" + RESEARCH_QUESTION + "/50_feature_norm.pickle"
            FEATURE_70 = "../pickle/" + RESEARCH_QUESTION + "/70_feature_norm.pickle"

            LABEL_20 = "../pickle/" + RESEARCH_QUESTION + "/20_label_norm.pickle"
            LABEL_50 = "../pickle/" + RESEARCH_QUESTION + "/50_label_norm.pickle"
            LABEL_70 = "../pickle/" + RESEARCH_QUESTION + "/70_label_norm.pickle"

            FEATURE_ALL = "../pickle/" + RESEARCH_QUESTION + "/all_feature_norm.pickle"
            LABEL_ALL = "../pickle/" + RESEARCH_QUESTION + "/all_label_norm.pickle"

        if RESEARCH_QUESTION == "q2":
            FEATURE_0 = "../pickle/" + RESEARCH_QUESTION + "/0_feature_norm.pickle"
            FEATURE_1 = "../pickle/" + RESEARCH_QUESTION + "/1_feature_norm.pickle"

            LABEL_0 = "../pickle/" + RESEARCH_QUESTION + "/0_label_norm.pickle"
            LABEL_1 = "../pickle/" + RESEARCH_QUESTION + "/1_label_norm.pickle"

            FEATURE_ALL = "../pickle/" + RESEARCH_QUESTION + "/all_feature_norm.pickle"
            LABEL_ALL = "../pickle/" + RESEARCH_QUESTION + "/all_label_norm.pickle"

        if RESEARCH_QUESTION == "q3":
            FEATURE_0 = "../pickle/" + RESEARCH_QUESTION + "/0_feature_norm.pickle"
            FEATURE_1 = "../pickle/" + RESEARCH_QUESTION + "/1_feature_norm.pickle"

            LABEL_0 = "../pickle/" + RESEARCH_QUESTION + "/0_label_norm.pickle"
            LABEL_1 = "../pickle/" + RESEARCH_QUESTION + "/1_label_norm.pickle"

            FEATURE_ALL = "../pickle/" + RESEARCH_QUESTION + "/all_feature_norm.pickle"
            LABEL_ALL = "../pickle/" + RESEARCH_QUESTION + "/all_label_norm.pickle"

    if RESEARCH_QUESTION == "q1":
        with open(FEATURE_20, "rb") as f:
            feature_20 = pickle.load(f)

        with open(FEATURE_50, "rb") as f:
            feature_50 = pickle.load(f)

        with open(FEATURE_70, "rb") as f:
            feature_70 = pickle.load(f)

        all_feature = np.concatenate((feature_20, feature_50, feature_70))

        # print(all_feature_norm)

        # all_label = read_pickle(LABEL_20, LABEL_50, LABEL_70)

        with open(LABEL_20, "rb") as f:
            label_20 = pickle.load(f)

        with open(LABEL_50, "rb") as f:
            label_50 = pickle.load(f)

        with open(LABEL_70, "rb") as f:
            label_70 = pickle.load(f)

        all_label = np.concatenate((label_20, label_50, label_70))

    if RESEARCH_QUESTION == "q2":
        with open(FEATURE_0, "rb") as f:
            feature_0 = pickle.load(f)

        with open(FEATURE_1, "rb") as f:
            feature_1 = pickle.load(f)

        all_feature = np.concatenate((feature_0, feature_1))

        # print(all_feature_norm)

        # all_label = read_pickle(LABEL_20, LABEL_50, LABEL_70)

        with open(LABEL_0, "rb") as f:
            label_0 = pickle.load(f)

        with open(LABEL_1, "rb") as f:
            label_1 = pickle.load(f)

        all_label = np.concatenate((label_0, label_1))

    if RESEARCH_QUESTION == "q3":
        with open(FEATURE_0, "rb") as f:
            feature_0 = pickle.load(f)

        with open(FEATURE_1, "rb") as f:
            feature_1 = pickle.load(f)

        all_feature = np.concatenate((feature_0, feature_1))

        # print(all_feature_norm)

        # all_label = read_pickle(LABEL_20, LABEL_50, LABEL_70)

        with open(LABEL_0, "rb") as f:
            label_0 = pickle.load(f)

        with open(LABEL_1, "rb") as f:
            label_1 = pickle.load(f)

        all_label = np.concatenate((label_0, label_1))

    all_feature_norm = zero_to_one(all_feature)

    with open(FEATURE_ALL, "wb") as f:
        pickle.dump(all_feature_norm, f, pickle.HIGHEST_PROTOCOL)

    with open(LABEL_ALL, "wb") as f:
        pickle.dump(all_label, f, pickle.HIGHEST_PROTOCOL)

    print("Data is normalized...")
