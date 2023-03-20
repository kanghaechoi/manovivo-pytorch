#! /bin/bash


run_svm() {
  QUESTION=$1
  OS=$2

  echo "Initiating $QUESTION SVM classification process on $OS..."

  while :
  do
    echo "Select a process? (Options: 1. feature_extraction, 2. data_normalize, 3. divide_dataset, 4. model training)"
    read PROCESS

    if [ $PROCESS = '1' ]
    then
      # Extract features to train SVM model
      echo "Input class name"
      read CLASS

      python ./src/feature_extraction.py $QUESTION $CLASS n

    elif [ $PROCESS = '2' ]
    then
      # Normalize dataset
      python ./src/data_normalize.py $QUESTION n
    elif [ $PROCESS = '3' ]
    then
      # Divide training and test dataset
      echo "How many features do you want to reduce?"
      read F_REDUCE
      python ./src/divide_dataset.py $QUESTION norm n $F_REDUCE
    elif [ $PROCESS = '4' ]
    then
      # Train SVM model and validate it
      python ./src/svm_model.py $QUESTION n
    else
      break
    fi
  done

  echo "Completed $QUESTION SVM classification process on $OS..."
}


run_knn() {
  QUESTION=$1
  OS=$2

  echo "Initiating $QUESTION KNN classification process on $OS..."

  while :
  do
    echo "Select a process? (Options: 1. feature_extraction, 2. data_normalize, 3. divide_dataset, 4. model training)"
    read PROCESS

    if [ $PROCESS = '1' ]
    then
      # Extract features to train SVM model
      python ./src/feature_extraction.py $QUESTION 20 n
      python ./src/feature_extraction.py $QUESTION 50 n
      python ./src/feature_extraction.py $QUESTION 70 n
    elif [ $PROCESS = '2' ]
    then
      # Normalize dataset
      python ./src/data_normalize.py $QUESTION n
    elif [ $PROCESS = '3' ]
    then
      # Divide training and test dataset
      echo "How many features do you want to reduce?"
      read F_REDUCE
      python ./src/divide_dataset.py $QUESTION norm n $F_REDUCE
    elif [ $PROCESS = '4' ]
    then
      # Train KNN model and validate it
      echo "How many neighbors do you want to compare?"
      read N_NEIGH
      python ./src/knn_model.py $QUESTION n $N_NEIGH
    else
      break
    fi
  done

  echo "Completed $QUESTION KNN classification process on $OS..."
}


run_rnn() {
  QUESTION=$1
  OS=$2

  echo "Initiating $QUESTION RNN classification process on $OS..."

  while :
  do
    echo "Select a process? (Options: 1. feature_extraction, 2. data_normalize, 3. divide_dataset, 4. model training)"
    read PROCESS

    if [ $PROCESS = '1' ]
    then
      # Extract features to train RNN model
      python ./src/feature_extraction_seq.py $QUESTION 20 n
      python ./src/feature_extraction_seq.py $QUESTION 50 n
      python ./src/feature_extraction_seq.py $QUESTION 70 n
    elif [ $PROCESS = '2' ]
    then
      # Combine dataset
      python ./src/join_seq_data.py $QUESTION n
    elif [ $PROCESS = '3' ]
    then
      # Divide training and test dataset
      echo "How many features do you want to reduce?"
      read F_REDUCE
      python ./src/divide_dataset.py $QUESTION seq n $F_F_REDUCE
    elif [ $PROCESS = '4' ]
    then
      # Train RNN model and validate it
      python ./src/lstm_model.py $QUESTION n
    else
      break
    fi
  done

  echo "Completed $QUESTION RNN classification process on $OS..."
}


run_cnn() {
  QUESTION=$1
  OS=$2

  echo "Initiating $QUESTION CNN classification process on $OS..."

  while :
  do
    echo "Select a process? (Options: 1. feature_extraction, 2. data_normalize, 3. divide_dataset, 4. model training)"
    read PROCESS

    if [ $PROCESS = '1' ]
    then
      # Extract features to train RNN model
      echo "Input the number of class"
      read CLASS_COUNT

      loop_count=0

      while [ $loop_count = $CLASS_COUNT ] :
      do
        echo "Input class name"
        read CLASS
        python .src/feature_extraction_seq.py $QUESTION $CLASS n

        loop_count=$((loop_count+1))
      done
    elif [ $PROCESS = '2' ]
    then
      # Combine dataset
      python .src/join_seq_data.py $QUESTION n
    elif [ $PROCESS = '3' ]
    then
      # Divide training and test dataset
      echo "How many features do you want to reduce?"
      read F_REDUCE
      python .src/divide_dataset.py $QUESTION seq n $F_REDUCE
    elif [ $PROCESS = '4' ]
    then
      # Train CNN model and validate it
      python .src/cnn_model.py $QUESTION n
    else
      break
    fi
  done

  echo "Completed $QUESTION CNN classification process on $OS..."
}


research_q1() {
  QUESTION=$1
  OS=$2

  echo "Q2-1. Which classifier do you want to use? (Options: svm, knn, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm $QUESTION $OS
  elif [ $selected_classifier = "knn" ]
  then
    run_knn $QUESTION $OS
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn $QUESTION $OS
  elif [ $selected_classifier = "cnn" ]
  then
    run_cnn $QUESTION $OS
  fi
}


research_q2() {
  QUESTION=$1
  OS=$2

  echo "Q2-2. Which classifier do you want to use? (Options: svm, knn, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm $QUESTION $OS
  elif [ $selected_classifier = "knn" ]
  then
    run_knn $QUESTION $OS
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn $QUESTION $OS
  elif [ $selected_classifier = "cnn" ]
  then
    run_cnn $QUESTION $OS
  fi
}


research_q3() {
  QUESTION=$1
  OS=$2

  echo "Q2-3. Which classifier do you want to use? (Options: svm, knn, rnn, or cnn)"
  read selected_classifier

  if [ $selected_classifier = "svm" ]
  then
    run_svm $QUESTION $OS
  elif [ $selected_classifier = "knn" ]
  then
    run_knn $QUESTION $OS
  elif [ $selected_classifier = "rnn" ]
  then
    run_rnn $QUESTION $OS
  elif [ $selected_classifier = "cnn" ]
  then
    run_cnn $QUESTION $OS
  fi
}

main() {
  echo "Q1. Which research QUESTION do you want to confirm? (Options: q1, q2, or q3)"
  read SELECTED_QUESTION

  UNIX="unix"

  if [ $SELECTED_QUESTION = "q1" ]
  then
    research_q1 $SELECTED_QUESTION $UNIX
  elif [ $SELECTED_QUESTION = "q2" ]
  then
    research_q2 $SELECTED_QUESTION $UNIX
  elif [ $SELECTED_QUESTION = "q3" ]
  then
    research_q3 $SELECTED_QUESTION $UNIX
  fi
}


main