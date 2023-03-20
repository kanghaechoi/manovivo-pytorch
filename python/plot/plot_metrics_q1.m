%% Initialize
clear;
clc;

%% Read data

q1_data = readmatrix('./q1/q1.txt');

iter_count = 3;
acc_idx = zeros(1, iter_count);
f1_idx = zeros(1, iter_count);

for i=1:iter_count
    acc_idx(1, i) = (2 * i) - 1;
end


for i=1:iter_count
    f1_idx(1, i) = (2 * i);
end

%% Q1 analysis without ReliefF

% SVM data
svm_idx = find(q1_data(:,1) == 1);

svm_data = q1_data(svm_idx, 3:end);
svm_acc = svm_data(:, acc_idx)';
svm_f1 = svm_data(:, f1_idx)';

q1_svm_no_rf = figure;
boxplot([svm_acc, svm_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 1 SVM result (without ReliefF)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q1_svm_no_rf,'./q1/q1_svm_no_rf','png')

% LSTM data
lstm_idx = find(q1_data(:,1) == 2);

lstm_data = q1_data(lstm_idx, 3:end);
lstm_acc = lstm_data(1, acc_idx)';
lstm_f1 = lstm_data(1, f1_idx)';

q1_lstm_no_rf = figure;
boxplot([lstm_acc, lstm_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 1 LSTM result (without ReliefF)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q1_lstm_no_rf,'./q1/q1_lstm_no_rf','png')

% CNN data
cnn_idx = find(q1_data(:,1) == 3);

cnn_data = q1_data(cnn_idx, 3:end);
cnn_acc = cnn_data(1, acc_idx)';
cnn_f1 = cnn_data(1, f1_idx)';

q1_cnn_no_rf = figure;
boxplot([cnn_acc, cnn_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 1 CNN result (without ReliefF)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q1_cnn_no_rf,'./q1/q1_cnn_no_rf','png')

%% Q1 analysis LSTM with ReliefF

% LSTM without 5 features data
lstm_idx = find(q1_data(:,1) == 2);

lstm_data = q1_data(lstm_idx, 3:end);
lstm_acc = lstm_data(2, acc_idx)';
lstm_f1 = lstm_data(2, f1_idx)';

q1_lstm_rf_5 = figure;
boxplot([lstm_acc, lstm_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 1 LSTM result (5 features are reduced)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q1_lstm_rf_5,'./q1/q1_lstm_rf_5','png')

% LSTM without 10 features data
lstm_idx = find(q1_data(:,1) == 2);

lstm_data = q1_data(lstm_idx, 3:end);
lstm_acc = lstm_data(2, acc_idx)';
lstm_f1 = lstm_data(2, f1_idx)';

q1_lstm_rf_10 = figure;
boxplot([lstm_acc, lstm_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 1 LSTM result (10 features are reduced)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q1_lstm_rf_10,'./q1/q1_lstm_rf_10','png')

%% Q1 analysis CNN with ReliefF

% CNN without 5 features data
cnn_idx = find(q1_data(:,1) == 3);

cnn_data = q1_data(cnn_idx, 3:end);
cnn_acc = cnn_data(2, acc_idx)';
cnn_f1 = cnn_data(2, f1_idx)';

q1_cnn_rf_5 = figure;
boxplot([cnn_acc, cnn_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 1 CNN result (5 features are reduced)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q1_cnn_rf_5,'./q1/q1_cnn_rf_5','png')

% CNN without 10 features data
cnn_idx = find(q1_data(:,1) == 3);

cnn_data = q1_data(cnn_idx, 3:end);
cnn_acc = cnn_data(2, acc_idx)';
cnn_f1 = cnn_data(2, f1_idx)';

q1_cnn_rf_10 = figure;
boxplot([cnn_acc, cnn_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 1 CNN result (10 features are reduced)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q1_cnn_rf_10,'./q1/q1_cnn_rf_10','png')
