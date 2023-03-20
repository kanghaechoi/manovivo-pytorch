import glob
import numpy as np
import pandas as pd
import pickle
import sys

from scipy import integrate
from sklearn import preprocessing


def get_data(path):
    raw_data = []
    raw_data_buffer = []
    prev_dir = 0
    move_count = 1

    with open(path, "r") as f:
        line_count = 0

        for line in f:
            line_list = line.split()
            curr_dir = float(line_list[0])

            if curr_dir > 0 and prev_dir < 0:
                raw_data.append(raw_data_buffer)
                move_count += 1
                raw_data_buffer = []
                raw_data_buffer.append(line_list)

                if move_count == 17:
                    break
            elif curr_dir < 0 and line_count == 0:
                continue
            else:
                raw_data_buffer.append(line_list)

            line_count += 1
            prev_dir = curr_dir

    return raw_data


def get_data_full(path):
    raw_data = []

    with open(path, "r") as f:
        for line in f:
            raw_data_buffer = line.split()
            raw_data.append(raw_data_buffer)

    raw_data = [raw_data]

    return raw_data


def zero_to_one(array):
    min_max_scaler = preprocessing.MinMaxScaler()
    array_transformed = min_max_scaler.fit_transform(array)

    return array_transformed


def mean_std(array):
    scale_processor = preprocessing.StandardScaler().fit(array)
    array_mean = scale_processor.mean_
    array_std = scale_processor.var_
    # array_time = scale_processor.n_samples_seen_

    return array_mean, array_std


def data_angle(input, col):
    angle_array = np.zeros((1, 2))
    input_dataframe = pd.DataFrame(input)[col]

    angle_temp = np.array(pd.to_numeric(input_dataframe))
    angle = angle_temp.reshape((-1, 1))
    # x_angle_transformed = zero_to_one(x_angle)
    # x_angle_mean, x_angle_std = mean_std(x_angle_transformed)
    angle_mean, angle_std = mean_std(angle)
    angle_array[0, 0] = angle_mean
    angle_array[0, 1] = angle_std

    return angle_array


def data_acc(input, col):
    acc_array = np.zeros((1, 2))
    input_dataframe = pd.DataFrame(input)[col]

    acc_temp = np.array(pd.to_numeric(input_dataframe))
    acc = acc_temp.reshape((-1, 1))
    # acc_transformed = zero_to_one(acc)
    # acc_mean, acc_std = mean_std(acc_transformed)
    acc_mean, acc_std = mean_std(acc)
    acc_array[0, 0] = acc_mean
    acc_array[0, 1] = acc_std

    return acc_array


def data_vel(input, col):
    vel_array = np.zeros((1, 2))
    input_dataframe = pd.DataFrame(input)[col]

    acc_temp = np.array(pd.to_numeric(input_dataframe))
    vel_temp = integrate.cumtrapz(acc_temp)
    vel = vel_temp.reshape((-1, 1))
    vel_mean, vel_std = mean_std(vel)

    vel_array[0, 0] = vel_mean
    vel_array[0, 1] = vel_std

    return vel_array


def data_time(hand, wrist):
    time_array = np.zeros((1, 1))

    max_time = max(len(hand), len(wrist))

    time_array[0, 0] = int(max_time)

    return time_array


def create_features(hand_data, wrist_data, helical_data):
    hand_x_angle = data_angle(hand_data, 0)
    hand_y_angle = data_angle(hand_data, 1)
    hand_z_angle = data_angle(hand_data, 2)

    thumb_x_angle = data_angle(hand_data, 3)
    index_x_angle = data_angle(hand_data, 4)

    thumb_x_acc = data_acc(hand_data, 6)
    thumb_x_vel = data_vel(hand_data, 6)
    thumb_y_acc = data_acc(hand_data, 7)
    thumb_y_vel = data_vel(hand_data, 7)
    thumb_z_acc = data_acc(hand_data, 8)
    thumb_z_vel = data_vel(hand_data, 8)

    index_x_acc = data_acc(hand_data, 9)
    index_x_vel = data_vel(hand_data, 9)
    index_y_acc = data_acc(hand_data, 10)
    index_y_vel = data_vel(hand_data, 10)
    index_z_acc = data_acc(hand_data, 11)
    index_z_vel = data_vel(hand_data, 11)

    wrist_x_angle = data_angle(wrist_data, 0)
    wrist_y_angle = data_angle(wrist_data, 1)
    wrist_z_angle = data_angle(wrist_data, 2)

    wrist_x_acc = data_acc(wrist_data, 3)
    wrist_x_vel = data_vel(wrist_data, 3)
    wrist_y_acc = data_acc(wrist_data, 4)
    wrist_y_vel = data_vel(wrist_data, 4)
    wrist_z_acc = data_acc(wrist_data, 5)
    wrist_z_vel = data_vel(wrist_data, 5)

    # time = data_time(hand, wrist)

    helical_x_angle = data_angle(helical_data, 0)
    helical_y_angle = data_angle(helical_data, 1)
    helical_z_angle = data_angle(helical_data, 2)

    feature_set = np.block(
        [
            hand_x_angle,
            hand_y_angle,
            hand_z_angle,
            thumb_x_angle,
            index_x_angle,
            thumb_x_acc,
            thumb_y_acc,
            thumb_z_acc,
            thumb_x_vel,
            thumb_y_vel,
            thumb_z_vel,
            index_x_acc,
            index_y_acc,
            index_z_acc,
            index_x_vel,
            index_y_vel,
            index_z_vel,
            wrist_x_angle,
            wrist_y_angle,
            wrist_z_angle,
            wrist_x_acc,
            wrist_y_acc,
            wrist_z_acc,
            wrist_x_vel,
            wrist_y_vel,
            wrist_z_vel,
            helical_x_angle,
            helical_y_angle,
            helical_z_angle,
        ]
    )

    return feature_set


def create_label(length, age):
    array_ones = np.ones((length, 1), dtype=int)
    age_labels = np.multiply(array_ones, age)

    return age_labels


if __name__ == "__main__":
    argument = sys.argv
    del argument[0]

    # RESEARCH_QUESTION = argument[0]
    # CLASS = argument[1]
    # IS_DEBUG = argument[2]

    RESEARCH_QUESTION = "q3"
    CLASS = "0"
    IS_DEBUG = "y"

    # print(RESEARCH_QUESTION)
    # print(CLASS)
    # print(IS_DEBUG)
    # exit(0)

    if IS_DEBUG == "n":
        if RESEARCH_QUESTION == "q1":
            FEATURE_PICKLE_PATH = (
                "./pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_feature_norm.pickle"
            )
            LABEL_PICKLE_PATH = (
                "./pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_label_norm.pickle"
            )

            path_hand = sorted(
                glob.glob("./data/" + RESEARCH_QUESTION + "/Hand_IMU_" + CLASS + "_*")
            )
            path_wrist = sorted(
                glob.glob("./data/" + RESEARCH_QUESTION + "/Wrist_IMU_" + CLASS + "_*")
            )
            path_helical = sorted(
                glob.glob(
                    "./data/" + RESEARCH_QUESTION + "/Helical_IMU_" + CLASS + "_*"
                )
            )

        if RESEARCH_QUESTION == "q2":
            FEATURE_PICKLE_PATH = (
                "./pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_feature_norm.pickle"
            )
            LABEL_PICKLE_PATH = (
                "./pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_label_norm.pickle"
            )

            path_hand = sorted(
                glob.glob(
                    "./data/" + RESEARCH_QUESTION + "/Hand_IMU_20_" + CLASS + "_*"
                )
            )
            path_wrist = sorted(
                glob.glob(
                    "./data/" + RESEARCH_QUESTION + "/Wrist_IMU_20_" + CLASS + "_*"
                )
            )
            path_helical = sorted(
                glob.glob(
                    "./data/" + RESEARCH_QUESTION + "/Helical_IMU_20" + CLASS + "_*"
                )
            )

        if RESEARCH_QUESTION == "q3":
            FEATURE_PICKLE_PATH = (
                "./pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_feature_norm.pickle"
            )
            LABEL_PICKLE_PATH = (
                "./pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_label_norm.pickle"
            )

            path_hand = sorted(
                glob.glob(
                    "./data/" + RESEARCH_QUESTION + "/Hand_IMU_20_" + CLASS + "_*"
                )
            )
            path_wrist = sorted(
                glob.glob(
                    "./data/" + RESEARCH_QUESTION + "/Wrist_IMU_20_" + CLASS + "_*"
                )
            )
            path_helical = sorted(
                glob.glob(
                    "./data/" + RESEARCH_QUESTION + "/Helical_IMU_20" + CLASS + "_*"
                )
            )

    if IS_DEBUG == "y":
        if RESEARCH_QUESTION == "q1":
            FEATURE_PICKLE_PATH = (
                "../pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_feature_norm.pickle"
            )
            LABEL_PICKLE_PATH = (
                "../pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_label_norm.pickle"
            )

            path_hand = sorted(
                glob.glob("../data/" + RESEARCH_QUESTION + "/Hand_IMU_" + CLASS + "_*")
            )
            path_wrist = sorted(
                glob.glob("../data/" + RESEARCH_QUESTION + "/Wrist_IMU_" + CLASS + "_*")
            )
            path_helical = sorted(
                glob.glob(
                    "../data/" + RESEARCH_QUESTION + "/Helical_IMU_" + CLASS + "_*"
                )
            )

        if RESEARCH_QUESTION == "q2":
            FEATURE_PICKLE_PATH = (
                "../pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_feature_norm.pickle"
            )
            LABEL_PICKLE_PATH = (
                "../pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_label_norm.pickle"
            )

            path_hand = sorted(
                glob.glob(
                    "../data/" + RESEARCH_QUESTION + "/Hand_IMU_20_" + CLASS + "_*"
                )
            )
            path_wrist = sorted(
                glob.glob(
                    "../data/" + RESEARCH_QUESTION + "/Wrist_IMU_20_" + CLASS + "_*"
                )
            )
            path_helical = sorted(
                glob.glob(
                    "../data/" + RESEARCH_QUESTION + "/Helical_IMU_20_" + CLASS + "_*"
                )
            )

        if RESEARCH_QUESTION == "q3":
            FEATURE_PICKLE_PATH = (
                "../pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_feature_norm.pickle"
            )
            LABEL_PICKLE_PATH = (
                "../pickle/" + RESEARCH_QUESTION + "/" + CLASS + "_label_norm.pickle"
            )

            path_hand = sorted(
                glob.glob(
                    "../data/" + RESEARCH_QUESTION + "/Hand_IMU_20_" + CLASS + "_*"
                )
            )
            path_wrist = sorted(
                glob.glob(
                    "../data/" + RESEARCH_QUESTION + "/Wrist_IMU_20_" + CLASS + "_*"
                )
            )
            path_helical = sorted(
                glob.glob(
                    "../data/" + RESEARCH_QUESTION + "/Helical_IMU_20_" + CLASS + "_*"
                )
            )

    subject_count = 0

    # aa = './data/Wrist_IMU_50_21.txt'
    #
    # wrist_ = get_data(aa)

    for hand, wrist, helical in zip(path_hand, path_wrist, path_helical):
        # print(hand)
        # print(wrist)
        list_idx = 0
        if RESEARCH_QUESTION == "q1":
            hand_lists = get_data(hand)
            wrist_lists = get_data(wrist)
            helical_lists = get_data(helical)

        if RESEARCH_QUESTION == "q2":
            hand_lists = get_data(hand)
            wrist_lists = get_data(wrist)
            helical_lists = get_data(helical)

        if RESEARCH_QUESTION == "q3":
            hand_lists = get_data_full(hand)
            wrist_lists = get_data_full(wrist)
            helical_lists = get_data_full(helical)

        for list_idx in range(
            min(len(hand_lists), len(wrist_lists), len(helical_lists))
        ):
            if list_idx > 0:
                feature_temp = create_features(
                    hand_lists[list_idx],
                    wrist_lists[list_idx],
                    helical_lists[list_idx],
                )
                feature = np.concatenate((feature, feature_temp))
            else:
                feature = create_features(
                    hand_lists[list_idx],
                    wrist_lists[list_idx],
                    helical_lists[list_idx],
                )

        if subject_count > 0:
            feature_set = np.concatenate((feature_set, feature))
        else:
            feature_set = feature

        subject_count += 1

    # print(feature_set.shape)
    # print(IS_DEBUG)
    # exit(0)

    with open(FEATURE_PICKLE_PATH, "wb") as f:
        pickle.dump(feature_set, f, pickle.HIGHEST_PROTOCOL)

    labels = create_label(feature_set.shape[0], int(CLASS))

    with open(LABEL_PICKLE_PATH, "wb") as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

    print(CLASS + "'s features are extracted...")
