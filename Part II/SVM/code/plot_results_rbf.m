% use only after svm_main.m has run.
if ~exist('results_rbf', 'var')
    load('./result_rbf.mat');
end

%% hyperparameters
% gamma_values = [0.001, 0.01, 0.1, 1];
% C_values = [0.1, 1, 10, 100];

% fine-tune round 1
% gamma_values = [0.01, 0.02, 0.04, 0.06, 0.08];
% C_values = [10, 20, 30, 40, 50, 60, 70, 80, 90];

% fine-tune round 2
gamma_values = [0.03, 0.035, 0.04, 0.045, 0.05];
C_values = [35, 40, 45, 50, 55];

% Soft-margin SVM accuracy
train_acc = results_rbf(:, 1:length(C_values));
test_acc = results_rbf(:, length(C_values)+1:end);

%% training accuracy
plot_acc = train_acc;
% Accuracy vs. gamma
figure;
plot(gamma_values, plot_acc, '-o', 'LineWidth', 2);
xlabel('gamma');
ylabel('Train Accuracy (%)');
legend(arrayfun(@(c) sprintf('C = %.1f', c), C_values, 'UniformOutput', false), 'Location', 'southwest');
title('Soft-Margin SVM: Train Accuracy vs. gamma');
grid on;
saveas(gcf, '../res_rbf/svm_accuracy_vs_gamma.png');

% Accuracy vs. C
figure;
plot(C_values, plot_acc', '-s', 'LineWidth', 2);
xlabel('Regularization Parameter (C)');
ylabel('Train Accuracy (%)');
legend(arrayfun(@(gamma) sprintf('gamma = %d', gamma), gamma_values, 'UniformOutput', false), 'Location', 'southwest');
title('Soft-Margin SVM: Train Accuracy vs. C');
grid on;
saveas(gcf, '../res_rbf/svm_accuracy_vs_C.png');

% Heatmap: accuracy vs. p,C 
figure;
imagesc(C_values, gamma_values, plot_acc);
colorbar;
xlabel('C');
ylabel('gamma');
title('Train Accuracy Heatmap');
xticks(C_values);
yticks(gamma_values);
set(gca, 'YDir', 'normal');
colormap('hot');
saveas(gcf, '../res_rbf/svm_accuracy_heatmap.png');
%% test accuracy

% Accuracy vs. gamma
figure;
plot(gamma_values, test_acc, '-o', 'LineWidth', 2);
xlabel('gamma');
ylabel('Test Accuracy (%)');
legend(arrayfun(@(c) sprintf('C = %.1f', c), C_values, 'UniformOutput', false), 'Location', 'southwest');
title('Soft-Margin SVM: Test Accuracy vs. gamma');
grid on;
saveas(gcf, '../res_rbf/test_svm_accuracy_vs_gamma.png');

% Accuracy vs. C
figure;
plot(C_values, test_acc', '-s', 'LineWidth', 2);
xlabel('Regularization Parameter (C)');
ylabel('Test Accuracy (%)');
legend(arrayfun(@(gamma) sprintf('gamma = %d', gamma), gamma_values, 'UniformOutput', false), 'Location', 'southwest');
title('Soft-Margin SVM: Test Accuracy vs. C');
grid on;
saveas(gcf, '../res_rbf/test_svm_accuracy_vs_C.png');

% Heatmap: accuracy vs. p,C 
figure;
imagesc(C_values, gamma_values, test_acc);
colorbar;
xlabel('C');
ylabel('gamma');
title('Test Accuracy Heatmap');
xticks(C_values);
yticks(gamma_values);
set(gca, 'YDir', 'normal');
colormap('hot');
saveas(gcf, '../res_rbf/test_svm_accuracy_heatmap.png');