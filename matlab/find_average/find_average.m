%% Initialize
clear;
clc;

%% Read file
fileID = fopen('./data/q3_lstm_bidir.txt','r');
% fileID = fopen('./data/q3_resnet_101.txt','r');
formatSpec = '%f %f %f %f';
matSize = [4, Inf];
f1_scores = fscanf(fileID,formatSpec,matSize);
fclose(fileID);

% f1_scores = f1_scores(:, 4);
f1_score_avg = mean(f1_scores);