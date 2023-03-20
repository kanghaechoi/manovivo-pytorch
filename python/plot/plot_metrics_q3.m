%% Initialize
clear;
clc;

%% Read data
q3_data = readmatrix('./q3/q3.txt');

iter_count = 3;
acc_idx = zeros(1, iter_count);
f1_idx = zeros(1, iter_count);

for i=1:iter_count
    acc_idx(1, i) = (2 * i) - 1;
end


for i=1:iter_count
    f1_idx(1, i) = (2 * i);
end

%% Q3 analysis LSTM with ReliefF

% LSTM with all features data
lstm_idx = find(q3_data(:,1) == 2);

lstm_data = q3_data(lstm_idx, 3:end);
lstm_acc = lstm_data(1, acc_idx)';
lstm_f1 = lstm_data(1, f1_idx)';

q3_lstm_no_rf = figure;
boxplot([lstm_acc, lstm_f1],'Labels',{'Accuracy','F1 score'})
ylim([70 100])
title('Question 3 LSTM result (without ReliefF)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q3_lstm_no_rf,'./q3/q3_lstm_no_rf','png')

%% Q3 analysis CNN with ReliefF

% CNN with all features data
cnn_idx = find(q3_data(:,1) == 3);

cnn_data = q3_data(cnn_idx, 3:end);
cnn_acc = cnn_data(1, acc_idx)';
cnn_f1 = cnn_data(1, f1_idx)';

q3_cnn_no_rf = figure;
boxplot([cnn_acc, cnn_f1],'Labels',{'Accuracy','F1 score'})
ylim([30 100])
title('Question 3 CNN result (without ReliefF)')
xlabel('Evaluation metrics')
ylabel('Performance')

saveas(q3_cnn_no_rf,'./q3/q3_cnn_no_rf','png')

