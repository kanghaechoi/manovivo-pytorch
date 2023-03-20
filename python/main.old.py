import os


def run_svm(q_num, OS):
    print(
        "Initiating " + q_num.upper() + " SVM classification process on " + OS + " ..."
    )

    while 1:
        process = input(
            "Select a process? (Options: 1. feature_extraction, "
            + "2. data_normalize, 3. divide_dataset, 4. model training)\n"
        )

        if process == "1":
            # Extract features to train SVM model
            CLASS = input("Input class name: ")

            os.system(
                "python ./src/feature_extraction.py " + q_num + " " + CLASS + " n"
            )
        elif process == "2":
            # Normalize dataset
            os.system("python ./src/data_normalize.py " + q_num + " n")
        elif process == "3":
            # Divide training and test dataset
            f_reduce = input("How many features do you want to reduce?\n")
            os.system("python ./src/divide_dataset.py " + q_num + " norm n " + f_reduce)
        elif process == "4":
            # Train SVM model and validate it
            os.system("python ./src/model_svm.py " + q_num + " n")
        else:
            break

    print(
        "Completed " + q_num.upper() + " SVM classification process on " + OS + " ...\n"
    )

    return 0


def run_knn(q_num, OS):
    print(
        "Initiating "
        + q_num.upper()
        + " KNN classification process on "
        + OS
        + " ...\n"
    )

    while 1:
        process = input(
            "Select a process? (Options: 1. feature_extraction, "
            + "2. data_normalize, 3. divide_dataset, 4. model training)\n"
        )

        if process == "1":
            # Extract features to train SVM model
            CLASS = input("Input class name: ")

            os.system(
                "python ./src/feature_extraction.py " + q_num + " " + CLASS + " n"
            )
        elif process == "2":
            # Normalize dataset
            os.system("python ./src/data_normalize.py " + q_num + " n")
        elif process == "3":
            # Divide training and test dataset
            f_reduce = input("How many features do you want to reduce?\n")
            os.system("python ./src/divide_dataset.py " + q_num + " norm n " + f_reduce)
        elif process == "4":
            # Train SVM model and validate it
            n_neigh = input("How many neighbors do you want to compare?\n")
            os.system("python ./src/model_knn.py " + q_num + " n " + n_neigh)
        else:
            break

    print(
        "Completed " + q_num.upper() + " KNN classification process on " + OS + " ...\n"
    )

    return 0


def run_rnn(q_num, OS):
    print(
        "Initiating "
        + q_num.upper()
        + " RNN classification process on "
        + OS
        + " ...\n"
    )

    while 1:
        process = input(
            "Select a process? (Options: 1. feature_extraction, "
            + "2. data_normalize, 3. divide_dataset, 4. model training)\n"
        )

        if process == "1":
            # Extract features to train RNN model
            CLASS = input("Input class name: ")

            os.system(
                "python ./src/feature_extraction_seq.py " + q_num + " " + CLASS + " n"
            )
        elif process == "2":
            # Combine dataset
            os.system("python ./src/join_seq_data.py " + q_num + " n")
        elif process == "3":
            # Divide training and test dataset
            f_reduce = input("How many features do you want to reduce?\n")
            os.system("python ./src/divide_dataset.py " + q_num + " seq n " + f_reduce)
        elif process == "4":
            # Train RNN model and validate it
            for i in range(6):
                print(
                    "-------------------------------------------Trial: "
                    + str(i + 1)
                    + "----------------------------------------"
                )
                # os.system("python ./src/model_lstm_single.py " + q_num + " n")
                os.system("python ./src/model_lstm_bidir.py " + q_num + " n")
        else:
            break

    print(
        "Completed " + q_num.upper() + " RNN classification process on " + OS + " ...\n"
    )

    return 0


def run_cnn(q_num, OS):
    print(
        "Initiating "
        + q_num.upper()
        + " CNN classification process on "
        + OS
        + " ...\n"
    )

    while 1:
        process = input(
            "Select a process? (Options: 1. feature_extraction, "
            + "2. data_normalize, 3. divide_dataset, 4. model training)\n"
        )

        if process == "1":
            # Extract features to train RNN model
            CLASS = input("Input class name: ")

            os.system(
                "python ./src/feature_extraction_seq.py " + q_num + " " + CLASS + " n"
            )
        elif process == "2":
            # Combine dataset
            os.system("python ./src/join_seq_data.py " + q_num + " n")
        elif process == "3":
            # Divide training and test dataset
            f_reduce = input("How many features do you want to reduce?\n")
            os.system("python ./src/divide_dataset.py " + q_num + " seq n " + f_reduce)
        elif process == "4":
            # Train CNN model and validate it
            # for i in range(3):
            # print(
            #     "-------------------------------------------Trial: "
            #     + str(i + 1)
            #     + "----------------------------------------"
            # )
            os.system("python ./src/model_resnet_50.py " + q_num + " n")
            # os.system("python ./src/model_resnet_101.py " + q_num + " n")
            # os.system("python ./src/model_resnet_152.py " + q_num + " n")
        else:
            break

    print(
        "Completed " + q_num.upper() + " CNN classification process on " + OS + " ...\n"
    )

    return 0


def research_q1(q_num, OS):
    selected_classifier = input(
        "Q2-1. Which classifier do you want to use? (Options: svm, knn, rnn, or cnn)\n"
    )

    if selected_classifier == "svm":
        run_svm(q_num, OS)

    if selected_classifier == "knn":
        run_knn(q_num, OS)

    if selected_classifier == "rnn":
        run_rnn(q_num, OS)

    if selected_classifier == "cnn":
        run_cnn(q_num, OS)

    return 0


def research_q2(q_num, OS):
    selected_classifier = input(
        "Q2-2. Which classifier do you want to use? (Options: svm, knn, rnn, or cnn)\n"
    )

    if selected_classifier == "svm":
        run_svm(q_num, OS)

    if selected_classifier == "knn":
        run_knn(q_num, OS)

    if selected_classifier == "rnn":
        run_rnn(q_num, OS)

    if selected_classifier == "cnn":
        run_cnn(q_num, OS)

    return 0


def research_q3(q_num, OS):
    selected_classifier = input(
        "Q2-3. Which classifier do you want to use? (Options: svm, knn, rnn, or cnn)\n"
    )

    if selected_classifier == "svm":
        run_svm(q_num, OS)

    if selected_classifier == "knn":
        run_knn(q_num, OS)

    if selected_classifier == "rnn":
        run_rnn(q_num, OS)

    if selected_classifier == "cnn":
        run_cnn(q_num, OS)

    return 0


if __name__ == "__main__":
    selected_question = input(
        "Q1. Which research question do you want to confirm? (Options: q1, q2, or q3)\n"
    )

    WINDOWS = str("windows")

    if selected_question == "q1":
        research_q1(selected_question, WINDOWS)

    if selected_question == "q2":
        research_q2(selected_question, WINDOWS)

    if selected_question == "q3":
        research_q3(selected_question, WINDOWS)
