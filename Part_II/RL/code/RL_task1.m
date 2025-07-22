set(0, 'DefaultFigureWindowStyle', 'docked');

%% Task 1
clear; clc; close all;
load task1.mat

folder_path = 'Task1/'
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
    fprintf('Folder "%s" has been created.\n', folder_path);
else
    fprintf('Folder "%s" already exists.\n', folder_path);
end

gamma_list = [0.5,0.9];
eps_list = {'1/k', '100/(100+k)', '(1+logk)/k', '(1+5logk)/k'};

results = {};

for i = 1:length(gamma_list)
    for j = 1:length(eps_list)
        gamma = gamma_list(i);
        eps_type = eps_list{j};

        fprintf('\nRunning: gamma=%.2f, eps_type=%s\n', gamma, eps_type);
        [opt_policy, success_runs, avg_time, opt_path, opt_reward] = Qlearning_main(reward, gamma, eps_type);

        results = [results; {gamma, eps_type, success_runs, avg_time}];

        if ~isempty(opt_policy)
            drawOptPol(opt_policy, [folder_path,sprintf('policy_g%.1f_eps%d.png', gamma, j)]);
            drawTraj(opt_path, opt_reward, [folder_path, sprintf('traj_g%.1f_eps%d.png', gamma, j)], reward);
            save([folder_path,sprintf('optPolicy_g%.1f_eps%d.mat', gamma, j)], 'opt_policy');
        end
    end
end

T = cell2table(results, ...
    'VariableNames', {'Gamma', 'EpsType', 'SuccessRuns', 'AvgTime'});
disp(T);
T.Gamma = cellfun(@(x) sprintf('%.4f', x), num2cell(T.Gamma), 'UniformOutput', false);
T.AvgTime = cellfun(@(x) sprintf('%.4f', x), num2cell(T.AvgTime), 'UniformOutput', false);
writetable(T, [folder_path,'Qlearning_summary.csv']);
fprintf('Summary table has been saved to Qlearning_summary.csv\n');


%% Task 1: Grid Search with selected eps_types and gamma values
load task1.mat

folder_path = 'Task1_gridSearch/';
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
    fprintf('Folder "%s" has been created.\n', folder_path);
else
    fprintf('Folder "%s" already exists.\n', folder_path);
end

gamma_list = [0.7,0.8,0.9];

eps_list = { ...
    '100/(100+k)', ...
    '(1+5logk)/k', ...
    '200/(200+k)', ...
    '300/(300+k)', ...
    '500/(500+k)', ...
};

results = {};

for i = 1:length(gamma_list)
    for j = 1:length(eps_list)
        gamma = gamma_list(i);
        eps_type = eps_list{j};

        fprintf('\nRunning: gamma = %.2f, eps_type = %s\n', gamma, eps_type);
        [opt_policy, success_runs, avg_time, opt_path, opt_reward] = ...
            Qlearning_main(reward, gamma, eps_type);

        results = [results; {gamma, eps_type, success_runs, avg_time}];

        if ~isempty(opt_policy)
            drawOptPol(opt_policy, [folder_path,sprintf('policy_g%.1f_eps%d.png', gamma, j)]);
            drawTraj(opt_path, opt_reward, [folder_path,sprintf('traj_g%.1f_eps%d.png', gamma, j)], reward);
            save([folder_path,sprintf('optPolicy_g%.1f_eps%d.mat', gamma, j)], 'opt_policy');
        end
    end
end

% Generate result table and export
T = cell2table(results, ...
    'VariableNames', {'Gamma', 'EpsType', 'SuccessRuns', 'AvgTime'});
disp(T);

% Format numbers
T.Gamma = cellfun(@(x) sprintf('%.4f', x), num2cell(T.Gamma), 'UniformOutput', false);
T.AvgTime = cellfun(@(x) sprintf('%.4f', x), num2cell(T.AvgTime), 'UniformOutput', false);
writetable(T, [folder_path,'Qlearning_summary_gridSearch.csv']);
fprintf('Summary table has been saved to Qlearning_summary_gridSearch.csv\n');