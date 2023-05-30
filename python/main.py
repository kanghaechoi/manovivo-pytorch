import numpy as np
import torch

from datasets.manovivo import Manovivo

from utilities.extraction import Extraction
from utilities.divide import Divide
from neural_networks.train.optimizer import Optimizer

from normalizations.min_max_normalization import MinMaxNormalization
from selections.relieff import ReliefF

from utilities.dimension import Dimension
from utilities.fetcher import Fetcher

from neural_networks.resnet.model import ResNet
from neural_networks.train.net_learning import NetLearning
from neural_networks.torch_device import torch_device


if __name__ == "__main__":
    print("======================")
    print("       MANOVIVO       ")
    print("======================")

    research_question = input(
        "Please select research question.\n(1) Research Question 1\n(2) Research Question 2\n(3) Research Question 3\nPress [Ctrl + C] to exit this program.\n"
    )
    research_question = int(research_question)

    manovivo = Manovivo(research_question, 100)
    manovivo.get_dataset(0)
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
    sample_length: int = 150  # 1 Second

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
    data, labels = extraction.extract_dataset()

    data_depth: int = data.shape[2]
    data_width: int = data.shape[1]
    data_height: int = data.shape[0]

    """
    Array dimension manipulation (Temporal)
    """
    dimension = Dimension()
    # if research_question == 2 or research_question == 3:
    data = dimension.numpy_squeeze(
        data,
        data_depth,
        data_width,
        data_height,
    )

    """
    Feature Normalization
    """
    normalization = MinMaxNormalization(data)
    normalized_data = normalization.transform(data)

    normalized_data = dimension.numpy_unsqueeze(
        normalized_data,
        data_width,
        data_height,
        data_depth,
    )

    divide = Divide(normalized_data, labels)
    divide.fit(test_dataset_ratio=0.2)

    training_data, training_labels = divide.training_dataset()
    test_data, test_labels = divide.test_dataset()

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
    training_data = np.expand_dims(training_data, axis=1)
    test_data = test_data[:, :, top_feature_indices]
    test_data = np.expand_dims(test_data, axis=1)

    """
    Neural Network
    """
    epochs = input("Please insert the number of epochs: \n")
    epochs: int = int(epochs)

    batch_size = input("Please insert batch size: \n")
    batch_size: int = int(batch_size)

    resnet_type = input(
        "Please select the number of ResNet layers (50 layers, 101 layers, 152 layers).\n"
    )
    resnet_type: int = int(resnet_type)

    if resnet_type == 50:
        resnet_block_parameters: list = [3, 4, 6, 3]
        model_weights_path: str = "./saved_weights/resnet_50_weights.pth"
    elif resnet_type == 101:
        resnet_block_parameters: list = [3, 4, 23, 3]
        model_weights_path: str = "./saved_weights/resnet_101_weights.pth"
    elif resnet_type == 152:
        resnet_block_parameters: list = [3, 8, 36, 3]
        model_weights_path: str = "./saved_weights/resnet_152_weights.pth"
    else:
        raise ValueError

    tensor = Tensor()

    resnet = ResNet(resnet_block_parameters, number_of_classes).to(tensor.device)

    optimizer = Optimizer(
        target="adam",
        network_parameters=resnet.parameters(),
        learning_rate=1e-4,
    )

    net_learning = NetLearning(model_weights_path)
    net_learning.train(
        model=resnet,
        data=torch.FloatTensor(training_data, device=torch_device),
        labels=torch.LongTensor(training_labels, device=torch_device),
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
    )

    net_learning.evaluate(
        neural_network=resnet,
        data=torch.FloatTensor(test_data, device=torch_device),
        labels=torch.LongTensor(test_labels, device=torch_device),
    )

    breakpoint()

    if KeyboardInterrupt:
        exit(0)
