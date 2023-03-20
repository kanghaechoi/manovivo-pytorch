%% Initialize
clear;
clc;

%% Read data
q2_data = readmatrix('./q2/q2.txt');

iter_count = 3;
acc_idx = zeros(1, iter_count);
f1_idx = zeros(1, iter_count);

for i=1:iter_count
    acc_idx(1, i) = (2 * i) - 1;
end


for i=1:iter_count
    f1_idx(1, i) = (2 * i);
end

%% Q2 analysis LSTM with ReliefF

% LSTM with all features data
lstm_idx = find(q2_data(:,1) == 2);

lstm_data = q2_data(lstm_idx, 3:end);
lstm_acc = lstm_data(1, acc_idx)';
lstm_f1 = lstm_data(1, f1_idx)';

q2_lstm_no_rf = figure;
boxplot([lstm_acc, lstm_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 2 LSTM result (without ReliefF)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q2_lstm_no_rf,'./q2/q2_lstm_no_rf','png')

% LSTM without 5 features data
lstm_idx = find(q2_data(:,1) == 2);

lstm_data = q2_data(lstm_idx, 3:end);
lstm_acc = lstm_data(2, acc_idx)';
lstm_f1 = lstm_data(2, f1_idx)';

q2_lstm_rf_5 = figure;
boxplot([lstm_acc, lstm_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 2 LSTM result (5 features are reduced)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q2_lstm_rf_5,'./q2/q2_lstm_rf_5','png')

% LSTM without 10 features data
lstm_idx = find(q2_data(:,1) == 2);

lstm_data = q2_data(lstm_idx, 3:end);
lstm_acc = lstm_data(2, acc_idx)';
lstm_f1 = lstm_data(2, f1_idx)';

q2_lstm_rf_10 = figure;
boxplot([lstm_acc, lstm_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 2 LSTM result (10 features are reduced)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q2_lstm_rf_10,'./q2/q2_lstm_rf_10','png')

%% Q2 analysis CNN with ReliefF

% CNN with all features data
cnn_idx = find(q2_data(:,1) == 3);

cnn_data = q2_data(cnn_idx, 3:end);
cnn_acc = cnn_data(1, acc_idx)';
cnn_f1 = cnn_data(1, f1_idx)';

q2_cnn_no_rf = figure;
boxplot([cnn_acc, cnn_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 2 CNN result (without ReliefF)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q2_cnn_no_rf,'./q2/q2_cnn_no_rf','png')

% CNN without 5 features data
cnn_idx = find(q2_data(:,1) == 3);

cnn_data = q2_data(cnn_idx, 3:end);
cnn_acc = cnn_data(2, acc_idx)';
cnn_f1 = cnn_data(2, f1_idx)';

q2_cnn_rf_5 = figure;
boxplot([cnn_acc, cnn_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 2 CNN result (5 features are reduced)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q2_cnn_rf_5,'./q2/q2_cnn_rf_5','png')

% CNN without 10 features data
cnn_idx = find(q2_data(:,1) == 3);

cnn_data = q2_data(cnn_idx, 3:end);
cnn_acc = cnn_data(2, acc_idx)';
cnn_f1 = cnn_data(2, f1_idx)';

q2_cnn_rf_10 = figure;
boxplot([cnn_acc, cnn_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 2 CNN result (10 features are reduced)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q2_cnn_rf_10, './q2/q2_cnn_rf_10', 'png')
