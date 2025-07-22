%% Task 3: Custom SVM (RBF Kernel)

% clearvars -except train_label train_data test_label test_data;
% if ~exist('train_label', 'var')
%     load('../data/train.mat');
%     load('../data/test.mat');
% end

%% Data Preprocessing
% Assume that 'train_data' 'train_label' 'test_label' 'test_data' already exist in the workspace

% Training set
[X_train, mu, sigma] = preprocess(train_data, 'train');
y_train = train_label;

% Test set
[X_test, ~, ~] = preprocess(test_data, 'test', mu, sigma);
y_test = test_label;

%% PCA (excluded in final design)
PCA=false;
if PCA
    disp('Dimensionality reduction via PCA');
    [coeff, score_train, ~, ~, explained, mu] = pca(X_train');
    
    k = 47;
    W_pca = coeff(:, 1:k);
    
    X_train_pca = (X_train' - mu) * W_pca;
    X_test_pca  = (X_test' - mu) * W_pca;
    
    X_train = X_train_pca';
    X_test = X_test_pca';
end
%% grid search
% Round 1
% gamma_values = [0.001, 0.01, 0.1, 1];
% C_values = [0.1, 1, 10, 100];

% Round 2
% gamma_values = [0.01, 0.02, 0.04, 0.06, 0.08];
% C_values = [10, 20, 30, 40, 50, 60, 70, 80, 90];

% Round 3
gamma_values = [0.03, 0.035, 0.04, 0.045, 0.05];
C_values = [35, 40, 45, 50, 55];

results_rbf = zeros(length(gamma_values), length(C_values)*2);
best_test_acc = 0;
best_train_acc = 0;
best_alpha = [];
best_b = [];
best_C = [];
best_gamma = [];
kernel_type = 'rbf';

for j = 1:length(gamma_values)
    gamma = gamma_values(j);
    for i = 1:length(C_values)
        C = C_values(i);
        
        % Train
        [alpha_custom_i, b_custom_i] = train_svm(X_train, y_train, C, kernel_type, gamma);

        % Predict
        y_pred_train = predict(X_train, y_train, X_train, alpha_custom_i, b_custom_i, kernel_type, gamma);
        train_acc = mean(y_pred_train == y_train);
        y_pred_test = predict(X_train, y_train, X_test, alpha_custom_i, b_custom_i, kernel_type, gamma);
        test_acc = mean(y_pred_test == y_test);

        results_rbf(j,i)=train_acc;
        results_rbf(j,i+length(C_values)) = test_acc;
        fprintf('gamma=%.3f, C=%.3f, Train acc=%.4f, Test acc=%.4f\n', gamma,C, train_acc, test_acc);
        
        if test_acc > best_test_acc
            best_test_acc = test_acc;
            best_train_acc = train_acc;
            best_alpha = alpha_custom_i;
            best_b = b_custom_i;
            best_gamma = gamma;
            best_C = C;
        end
    end
end

fprintf('\nBest RBF SVM: gamma=%.3f, C=%.3f, Train Accuracy=%.4f, Test Accuracy=%.4f\n', ...
        best_gamma,best_C,  best_train_acc, best_test_acc);

save('result_rbf.mat', "results_rbf", '-mat');

%% Evaluation with eval_data
if exist('eval_data', 'var')
    eval_predicted = predict(X_train, y_train, eval_data, best_alpha, best_b, kernel_type, best_gamma);
end