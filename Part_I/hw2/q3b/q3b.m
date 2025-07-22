clc
clear
close all

% load dataset
img_path= '../Data.mat';
load(img_path);

%% Manual Global Normalization
[train_mean, train_std] = global_mean_variance(X_train);

% Apply normalization using global statistics
X_norm_train = (X_train - train_mean) / train_std;
X_norm_test = (X_test - train_mean) / train_std;

% Verify Normalization Effectiveness (Expected: mean≈0, variance≈1)
disp(['Normalized Mean: ', num2str(mean(X_norm_train(:)))]);
disp(['Normalized Variance: ', num2str(var(X_norm_train(:)))]);

% Comparative Experiment with Identical Perceptron Parameters
[net_norm, tr_norm] = perceptron_train(X_norm_train, Y_train, X_norm_test, Y_test); % Manual global normalization

%% Feature-wise Standardization
[X_train_z, X_test_z] = safe_zscore(X_train, X_test);
[net_zscore, tr_zscore] = perceptron_train(X_train_z, Y_train, X_test_z, Y_test);




%% Calculate Dataset Global Statistics
function [global_mean, global_std] = global_mean_variance(X)
% Calculate global statistics across all samples and features
% Input:
%   X - Data matrix (features × samples)
% Output:
%   global_mean - Global mean of all feature values
%   global_std  - Population standard deviation of all feature values

global_mean = mean(X(:)); % Ensemble mean across entire dataset
global_std = std(X(:));   % Population standard deviation (N normalization)
end

%% Robust Feature Standardization with Zero-Variance Protection
function [X_train_z, X_test_z, mu, sigma] = safe_zscore(X_train, X_test)
% Perform z-score standardization using training statistics
% Inputs:
%   X_train - Training data matrix (features × samples)
%   X_test  - Test data matrix (features × samples)
% Outputs:
%   X_train_z - Standardized training data (μ=0, σ=1 per feature)
%   X_test_z  - Standardized test data (using training μ/σ)
%   mu        - Feature-wise means computed from training data
%   sigma     - Feature-wise standard deviations with zero-variance protection

% Compute training statistics
mu = mean(X_train, 2);         % Column-wise mean (per feature)
sigma = std(X_train, 0, 2);    % Population std (N normalization)

% Handle zero-variance features to prevent division errors
sigma(sigma == 0) = 1;         % Apply unit variance to invariant features

% Standardize data using training parameters
X_train_z = (X_train - mu) ./ sigma; % Center and scale training data
X_test_z = (X_test - mu) ./ sigma;   % Apply same transformation to test data
end


