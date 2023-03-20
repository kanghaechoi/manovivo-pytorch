import pickle
import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix


def load_data(feature_path, label_path):
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)
    with open(label_path, "rb") as f:
        label = pickle.load(f)

    return feature, label


if __name__ == "__main__":
    argument = sys.argv
    del argument[0]

    RESEARCH_QUESTION = argument[0]
    IS_DEBUG = argument[1]

    if IS_DEBUG == "n":
        TRAIN_FEATURE_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/train_feature_norm.pickle"
        )
        TEST_FEATURE_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/test_feature_norm.pickle"
        )

        TRAIN_LABEL_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/train_label_norm.pickle"
        )
        TEST_LABEL_PATH = (
            "./pickle/" + RESEARCH_QUESTION + "/test_label_norm.pickle"
        )

    if IS_DEBUG == "y":
        TRAIN_FEATURE_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/train_feature_norm.pickle"
        )
        TEST_FEATURE_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/test_feature_norm.pickle"
        )

        TRAIN_LABEL_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/train_label_norm.pickle"
        )
        TEST_LABEL_PATH = (
            "../pickle/" + RESEARCH_QUESTION + "/test_label_norm.pickle"
        )

    train_feature, train_label = load_data(TRAIN_FEATURE_PATH, TRAIN_LABEL_PATH)
    test_feature, test_label = load_data(TEST_FEATURE_PATH, TEST_LABEL_PATH)

    svm_model = LinearSVC(tol=1e-5)
    svm_model.fit(train_feature, np.squeeze(train_label))

    predicted_label = svm_model.predict(test_feature).reshape((-1, 1))
    test_len = len(predicted_label)

    err_array = np.subtract(predicted_label, test_label)
    err_idx = np.where(err_array != 0)[1]

    err = round(((len(err_idx) / test_len) * 100), 2)
    acc = 100 - err

    f1 = f1_score(test_label, predicted_label, average="macro")
    f1 = round((f1 * 100), 2)

    conf_mat = confusion_matrix(test_label, predicted_label)

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

        len_20 = np.where(test_label == 20)[0].size
        len_50 = np.where(test_label == 50)[0].size
        len_70 = np.where(test_label == 70)[0].size

        accu = (conf_mat_11 + conf_mat_22 + conf_mat_33) / np.sum(
            conf_mat.ravel()
        )
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

    print("SVM model's accuracy is ", acc, "%")
    print("SVM model's F1 score is ", f1, "%")
    print(accu)
    print(specificity)
    print(recall)
    print(precision)
    print(f1_s)
    print(far)
    print(frr)
