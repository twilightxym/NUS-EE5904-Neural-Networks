%% Task 1 and Task 2

%clearvars -except train_label train_data test_label test_data

%% Load data
% load('../data/train.mat');
% load('../data/test.mat');

%% Data Preprocessing

% Training set
[X_train, mu, sigma] = preprocess(train_data, 'train');
y_train = train_label;

% Test set
[X_test, ~, ~] = preprocess(test_data, 'test', mu, sigma);
y_test = test_label;

%% Train SVMs
% Hard-margin linear SVM
C_hard = 1e6;
[alpha_linear, b_linear] = train_svm(X_train, y_train, C_hard, 'linear', 1, true);

% Hard-margin polynomial SVM (p=2 to 5)
p_values = 2:5;
alpha_poly = cell(1, 4);
b_poly = cell(1, 4);
for i = 1:4
    [alpha_poly{i}, b_poly{i}] = train_svm(X_train, y_train, C_hard, 'poly', p_values(i), true);
end

% Soft-margin polynomial SVM (C=0.1,0.6,1.1,2.1)
C_values = [0.1, 0.6, 1.1, 2.1];
alpha_soft = cell(5, 4);
b_soft = cell(5, 4);
for p = 1:5
    for C = 1:4
        [alpha_soft{p,C}, b_soft{p,C}] = train_svm(X_train, y_train, C_values(C), 'poly', p, true);
    end
end

%% Evaluate performance
% Initialize results table
results = zeros(7, 8); % Table format per specifications

% Hard-margin linear SVM
y_pred_train = predict(X_train, y_train, X_train, alpha_linear, b_linear, 'linear', 1);
y_pred_test = predict(X_train, y_train, X_test, alpha_linear, b_linear, 'linear', 1);
results(1, [1 5]) = [mean(y_pred_train == y_train), mean(y_pred_test == y_test)];

% Hard-margin polynomial SVM (p=2 to 5)
for i = 1:4
    current_p = p_values(i);

    y_pred_train = predict(X_train, y_train, X_train, alpha_poly{i}, b_poly{i}, 'poly', current_p);
    train_acc = mean(y_pred_train == y_train);

    y_pred_test = predict(X_train, y_train, X_test, alpha_poly{i}, b_poly{i}, 'poly', current_p);
    test_acc = mean(y_pred_test == y_test);

    results(2, i) = train_acc;      % row 2，col 1-4
    results(2, 4 + i) = test_acc;   % row 2，col 5-8
end

% Soft-margin polynomial SVM (p=1 to 5, C=0.1,0.6,1.1,2.1)
for p = 1:5
    for C = 1:4
        y_pred_train = predict(X_train, y_train, X_train, alpha_soft{p,C}, b_soft{p,C}, 'poly', p);
        train_acc = mean(y_pred_train == y_train);
    
        y_pred_test = predict(X_train, y_train, X_test, alpha_soft{p,C}, b_soft{p,C}, 'poly', p);
        test_acc = mean(y_pred_test == y_test);
    
        results(2+p, C) = train_acc;      % row 3-7，col 1-4
        results(2+p, 4 + C) = test_acc;   % row 3-7，col 5-8
    end
end

save('result.mat', "results", '-mat');