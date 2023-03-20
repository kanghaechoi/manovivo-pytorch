%% Initialize
clear;
clc;

%% Read file
fileID = fopen('f1_scores.txt','r');
formatSpec = '%f %f %f';
matSize = [3, Inf];
f1_scores = fscanf(fileID, formatSpec, matSize);
fclose(fileID);

x = [185, 922, 2032];
y1 = fliplr(f1_scores(:, 1)');
y2 = fliplr(f1_scores(:, 2)');
y3 = fliplr(f1_scores(:, 3)');
y4 = fliplr(f1_scores(:, 4)');
y5 = fliplr(f1_scores(:, 5)');
y6 = fliplr(f1_scores(:, 6)');
y7 = fliplr(f1_scores(:, 7)');
y8 = fliplr(f1_scores(:, 8)');
y9 = fliplr(f1_scores(:, 9)');

%% Plot a graph

plot(x, y1, '-s', 'LineWidth', 1, 'Color', '#0072BD')
title('F1 Score Comparison Among Models','FontSize',14,'FontWeight','bold')
xlabel('The Size of Dataset','FontWeight','bold')
ylabel('F1 Score','FontWeight','bold')
hold on
plot(x, y2, '-s', 'LineWidth', 1, 'Color', '#D95319')
hold on
plot(x, y3, '-.^', 'LineWidth', 1, 'Color', '#D95319')
hold on
plot(x, y4, '--d', 'LineWidth', 1, 'Color', '#D95319')
hold on
plot(x, y5, '-s', 'LineWidth', 1, 'Color', '#7E2F8E')
hold on
plot(x, y6, '-.*', 'LineWidth', 1, 'Color', '#7E2F8E')
hold on
plot(x, y7, '-s', 'LineWidth', 1, 'Color', '#4DBEEE')
hold on
plot(x, y8, '-.^', 'LineWidth', 1, 'Color', '#4DBEEE')
hold on
plot(x, y9, '--*', 'LineWidth', 1, 'Color', '#4DBEEE')
hold off

legend({'SVM','k-NN (k=5)','k-NN (k=10)','k-NN (k=15)', ...
    'LSTM','Bi-LSTM','ResNet-50','ResNet-101','ResNet-152'}, ...
    'Location', 'northeastoutside');

% xticks([100 185 200 300 4 5 6 7 ])
% xticklabels({'2032','922','185'})
yticks([80 82.5 85 87.5 90 92.5 95 97.5 100])
yticklabels({'80','82.5','85','87.5','90','92.5','95','97.5','100'})
ytickformat('percentage')
xlim([0 2200])
ylim([79 101])

ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.GridLineStyle = ':';
ax.GridAlpha = 0.5;

