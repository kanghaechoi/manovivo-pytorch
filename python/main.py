import numpy as np
import torch.optim as optim

from dataset.extraction import Extraction
from dataset.divide import Divide

from filters.normalization import MinMaxNormalization
from filters.relieff import ReliefF

from utilities.dimension import Dimension
from utilities.fetcher import Fetcher

from neural_networks.resnet import ResNet
from neural_networks.net_learning import NetLearning


if __name__ == "__main__":
    print("======================")
    print("       MANOVIVO       ")
    print("======================")

    research_question = input(
        "Please select research question.\n(1) Research Question 1\n(2) Research Question 2\n(3) Research Question 3\nPress [Ctrl + C] to exit this program.\n"
    )
    research_question = int(research_question)

    """
    Data fetch configuration
    """
    if research_question == 1:
        subject_age_1: int = 20
        subject_age_2: int = 50
        subject_age_3: int = 70

        ages = [subject_age_1, subject_age_2, subject_age_3]

        authentication_classes = None
        authentication_flag: bool = False
        number_of_classes = len(ages)
    elif research_question == 2 or research_question == 3:
        subject_age_1 = 20

        is_not_authorized: int = 0
        is_authorized: int = 1

        ages = [subject_age_1]
        authentication_classes = [is_authorized, is_not_authorized]
        authentication_flag: bool = True
        number_of_classes = len(authentication_classes)
    else:
        raise ValueError

    """
    Data fetch
    """
    # sample_length = input("Please insert sample length. (1 sample length = 1/100 sec)")
    # sample_length = int(sample_length)
    # sample_length: int = 50  # 0.5 Seconds
    sample_length: int = 100  # 1 Second

    fetch = Fetcher(
        research_question,
        ages,
        sample_length,
        authentication_flag,
        authentication_classes,
    )
    data_chunks: zip = fetch.fetch_dataset_chunks()

    extraction = Extraction(
        research_question,
        ages,
        data_chunks,
        sample_length,
        authentication_classes,
    )
    _data, _labels = extraction.extract_dataset()

    data_depth: int = _data.shape[2]
    data_width: int = _data.shape[1]
    data_height: int = _data.shape[0]

    """
    Array dimension manipulation (Temporal)
    """
    dimension = Dimension()
    # if research_question == 2 or research_question == 3:
    _data = dimension.numpy_squeeze(
        _data,
        data_depth,
        data_width,
        data_height,
    )

    """
    Feature Normalization
    """
    normalization = MinMaxNormalization(_data)
    normalized_data = normalization.transform(_data)

    normalized_data = dimension.numpy_unsqueeze(
        normalized_data,
        data_depth,
        data_width,
        data_height,
    )

    divide = Divide(normalized_data, _labels)
    divide.fit(test_dataset_ratio=0.2)

    training_data, training_labels = divide.training_dataset()
    data, labels = divide.test_dataset()

    """
    Feature selection using ReliefF algorithm
    """
    number_of_feature_reduction = input(
        "Please insert number of features to be reduced.\n"
    )
    number_of_feature_reduction = int(number_of_feature_reduction)

    if research_question == 2 or research_question == 3:
        training_data_for_relieff = np.sum(training_data, axis=1)

    relieff = ReliefF(
        n_neighbors=normalized_data.shape[2],
        n_features_to_keep=normalized_data.shape[2] - number_of_feature_reduction,
    )
    relieff.fit_transform(training_data_for_relieff, np.squeeze(training_labels))
    top_feature_indices = relieff.top_features[0 : relieff.n_features_to_keep]

    training_data = training_data[:, :, top_feature_indices]
    training_data = np.expand_dims(training_data, axis=3)
    data = data[:, :, top_feature_indices]
    data = np.expand_dims(data, axis=3)

    """
    Model training
    """
    epochs = input("Please insert the number of epochs: ")
    epochs: int = int(epochs)

    batch_size = input("Please insert batch size: ")
    batch_size: int = int(batch_size)

    resnet_type = input(
        "Please select a model to train.\n(1) ResNet-50\n(2) ResNet-101\n(3) ResNet-152\n"
    )
    resnet_type: int = int(resnet_type)

    if resnet_type == 1:
        resnet_block_parameters: list = [3, 4, 6, 3]
        save_path: str = "./python/saved_models/resnet50"
    elif resnet_type == 2:
        resnet_block_parameters: list = [3, 4, 23, 3]
        save_path: str = "./python/saved_models/resnet101"
    elif resnet_type == 3:
        resnet_block_parameters: list = [3, 8, 36, 3]
        save_path: str = "./python/saved_models/resnet152"
    else:
        raise ValueError

    resnet = ResNet(resnet_block_parameters, number_of_classes)

    adam_optimizer = optim.Adam([], lr=0.0001)
    rms_prop_optimizer = optim.RMSprop([], lr=0.0001)

    net_learning = NetLearning(save_path)
    net_learning.train(
        _model=resnet,
        _data=training_data,
        _labels=training_labels,
        _epochs=epochs,
        _batch_size=batch_size,
        _optimizer=adam_optimizer,
    )

    net_learning.test(_model=resnet, _data=data, _labels=labels)

    breakpoint()

    if KeyboardInterrupt:
        exit(0)
