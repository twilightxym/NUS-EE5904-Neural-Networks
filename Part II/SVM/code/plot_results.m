% use only after main_workflow.m has run.
if ~exist('results', 'var')
    load('./result.mat');
end

%% hyperparameters
p_values = [1, 2, 3, 4, 5];
C_values = [0.1, 0.6, 1.1, 2.1];

%% Soft-margin SVM accuracy
train_acc = results(3:7, 1:4);
test_acc = results(3:7, 5:8);

plot_acc = train_acc;
% Accuracy vs. p
figure;
plot(p_values, plot_acc, '-o', 'LineWidth', 2);
xlabel('Polynomial Degree (p)');
ylabel('Train Accuracy (%)');
legend(arrayfun(@(c) sprintf('C = %.1f', c), C_values, 'UniformOutput', false), 'Location', 'southwest');
title('Soft-Margin SVM: Train Accuracy vs. p');
grid on;
saveas(gcf, '../res/svm_accuracy_vs_p.png');

% Accuracy vs. C
figure;
plot(C_values, plot_acc', '-s', 'LineWidth', 2);
xlabel('Regularization Parameter (C)');
ylabel('Train Accuracy (%)');
legend(arrayfun(@(p) sprintf('p = %d', p), p_values, 'UniformOutput', false), 'Location', 'southwest');
title('Soft-Margin SVM: Train Accuracy vs. C');
grid on;
saveas(gcf, '../res/svm_accuracy_vs_C.png');

% Heatmap: accuracy vs. p,C 
figure;
imagesc(C_values, p_values, plot_acc);
colorbar;
xlabel('C');
ylabel('Polynomial Degree p');
title('Train Accuracy Heatmap');
xticks(C_values);
yticks(p_values);
set(gca, 'YDir', 'normal');
colormap('hot');
saveas(gcf, '../res/svm_accuracy_heatmap.png');