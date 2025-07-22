%% RL_main.m - Task 2: Final Evaluation with Selected Parameters 
clear; clc; close all;

% Assuming you have loaded reward matrix (qevalreward)
load qeval.mat

% Ensure qevalreward is available in workspace
if ~exist('qevalreward', 'var')
    error('Variable "qevalreward" is not found in workspace. Please load qeval.mat before running.');
end

% Create folder for saving results
folder_path = 'Task2/';
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
    fprintf('Folder "%s" has been created.\n', folder_path);
else
    fprintf('Folder "%s" already exists.\n', folder_path);
end

% Selected parameters from Grid Search
gamma = 0.7;
eps_type = '500/(500+k)';
fprintf('Running final Q-learning with gamma = %.2f, eps_type = %s\n', gamma, eps_type);

% Run Q-learning using selected parameters
[opt_policy, success_runs, avg_time, opt_path, opt_reward] = Qlearning_main(qevalreward, gamma, eps_type);

% Display summary
fprintf('\n==== Final Q-learning Result ====\n');
fprintf('Gamma = %.4f\n', gamma);
fprintf('Epsilon Type = %s\n', eps_type);
fprintf('Successful Runs = %d / 10\n', success_runs);
fprintf('Average Time = %.4f seconds\n', avg_time);
fprintf('Total Discounted Reward = %.4f\n', opt_reward);

% Required output: qevalstates as a column vector
qevalstates = opt_path(:);
assignin('base', 'qevalstates', qevalstates); % Save to workspace as required

% Save optimal policy and path visualizations
drawOptPol(opt_policy, [folder_path,'task2_optimal_policy.png']);
drawTraj(opt_path, opt_reward, [folder_path,'task2_optimal_trajectory.png'], qevalreward);
save([folder_path,'task2_optPolicy.mat'], 'opt_policy');
fprintf('Visualization and optimal policy saved to %s\n', folder_path);

% Display total reward
disp(['Total Reward: ', num2str(opt_reward)]);
disp('qevalstates: ');
disp(qevalstates);