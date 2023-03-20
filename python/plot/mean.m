clear;
clc;

data = readmatrix('./resnet152/q3/0/result.txt');

data_ = data(:, 2:end);

data_mean = sum(data_, 2)/3;