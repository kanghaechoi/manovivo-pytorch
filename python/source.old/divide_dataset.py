import pickle
import numpy as np
import math
import random
import sys
from ReliefF import ReliefF
from sklearn import preprocessing


def write_train_test(train_path, train_data, test_path, test_data):
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

    return 0


if __name__ == '__main__':
    argument = sys.argv
    del argument[0]

    RESEARCH_QUESTION = argument[0]
    TARGET = argument[1]
    IS_DEBUG = argument[2]
    F_REDUCE = argument[3]

    # RESEARCH_QUESTION = 'q1'
    # TARGET = 'seq'
    # IS_DEBUG = 'y'
    # F_REDUCE = 0

    if(IS_DEBUG == 'n'):
        if(TARGET == 'norm'):
            ALL_FEATURE = './pickle/' + RESEARCH_QUESTION + '/all_feature_norm.pickle'
            ALL_LABEL = './pickle/' + RESEARCH_QUESTION + '/all_label_norm.pickle'

            TRAIN_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/train_feature_norm.pickle'
            TEST_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/test_feature_norm.pickle'

            TRAIN_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/train_label_norm.pickle'
            TEST_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/test_label_norm.pickle'

            SHAPE_IDX = 0

        if(TARGET == 'seq'):
            ALL_FEATURE = './pickle/' + RESEARCH_QUESTION + '/all_feature_seq.pickle'
            ALL_LABEL = './pickle/' + RESEARCH_QUESTION + '/all_label_seq.pickle'

            TRAIN_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/train_feature_seq.pickle'
            TEST_FEATURE_PATH = './pickle/' + RESEARCH_QUESTION + '/test_feature_seq.pickle'

            TRAIN_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/train_label_seq.pickle'
            TEST_LABEL_PATH = './pickle/' + RESEARCH_QUESTION + '/test_label_seq.pickle'

            SHAPE_IDX = 0

    if (IS_DEBUG == 'y'):
        if (TARGET == 'norm'):
            ALL_FEATURE = '../pickle/' + RESEARCH_QUESTION + '/all_feature_norm.pickle'
            ALL_LABEL = '../pickle/' + RESEARCH_QUESTION + '/all_label_norm.pickle'

            TRAIN_FEATURE_PATH = '../pickle/' + RESEARCH_QUESTION + '/train_feature_norm.pickle'
            TEST_FEATURE_PATH = '../pickle/' + RESEARCH_QUESTION + '/test_feature_norm.pickle'

            TRAIN_LABEL_PATH = '../pickle/' + RESEARCH_QUESTION + '/train_label_norm.pickle'
            TEST_LABEL_PATH = '../pickle/' + RESEARCH_QUESTION + '/test_label_norm.pickle'

            SHAPE_IDX = 0

        if (TARGET == 'seq'):
            ALL_FEATURE = '../pickle/' + RESEARCH_QUESTION + '/all_feature_seq.pickle'
            ALL_LABEL = '../pickle/' + RESEARCH_QUESTION + '/all_label_seq.pickle'

            TRAIN_FEATURE_PATH = '../pickle/' + RESEARCH_QUESTION + '/train_feature_seq.pickle'
            TEST_FEATURE_PATH = '../pickle/' + RESEARCH_QUESTION + '/test_feature_seq.pickle'

            TRAIN_LABEL_PATH = '../pickle/' + RESEARCH_QUESTION + '/train_label_seq.pickle'
            TEST_LABEL_PATH = '../pickle/' + RESEARCH_QUESTION + '/test_label_seq.pickle'

            SHAPE_IDX = 0

    with open(ALL_FEATURE, 'rb') as f:
        all_feature = pickle.load(f)

    with open(ALL_LABEL, 'rb') as f:
        all_label = pickle.load(f)

    all_idx = np.linspace(0, (all_feature.shape[SHAPE_IDX] - 1), num=all_feature.shape[SHAPE_IDX], dtype=int)

    random.shuffle(all_idx)

    if(TARGET == 'norm'):
        all_feature = all_feature[all_idx, :]
    elif(TARGET == 'seq'):
        all_feature = all_feature[all_idx, :, :]

    all_label = all_label[all_idx, :]

    test_len = math.floor(all_feature.shape[SHAPE_IDX] * 0.2)

    test_idx = random.sample(all_idx.tolist(), test_len)

    train_idx = np.delete(all_idx.reshape((1, -1)), test_idx).tolist()

    if(TARGET == 'norm'):
        train_feature = all_feature[train_idx, :]
        test_feature = all_feature[test_idx, :]
    if(TARGET == 'seq'):
        train_feature = all_feature[train_idx, :, :]
        test_feature = all_feature[test_idx, :, :]

    train_label = all_label[train_idx, :]
    test_label = all_label[test_idx, :]

    F_REDUCE = int(F_REDUCE)
    if(F_REDUCE != 0):

        fs = ReliefF(n_neighbors=all_feature.shape[1], n_features_to_keep=(all_feature.shape[1] - F_REDUCE))

        if(TARGET == 'norm'):
            train_feature = fs.fit_transform(train_feature, np.squeeze(train_label))
            test_feature = fs.transform(test_feature)
        if(TARGET == 'seq'):
            train_feature_sum = np.sum(train_feature, axis=1)
            fs.fit_transform(train_feature_sum, np.squeeze(train_label))
            top_feature_idx = fs.top_features[0:(all_feature.shape[2] - F_REDUCE)]
            train_feature = train_feature[:, :, top_feature_idx]
            test_feature = test_feature[:, :, top_feature_idx]

    write_train_test(TRAIN_FEATURE_PATH, train_feature, TEST_FEATURE_PATH, test_feature)

    write_train_test(TRAIN_LABEL_PATH, train_label, TEST_LABEL_PATH, test_label)

    print('Divided train dataset and test dataset...')