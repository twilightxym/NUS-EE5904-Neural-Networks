% Visualize epsilon decay curves from different expressions
clear; clc; close all;

k = 1:2000;

% Compute epsilon values for each expression
eps_1_over_k        = 1 ./ k;
eps_100_over_100k   = 100 ./ (100 + k);
eps_1log_over_k     = (1 + log(k)) ./ k;
eps_1_5log_over_k   = (1 + 5 * log(k)) ./ k;

% Find k where epsilon drops below 0.05
k_1_over_k        = find(eps_1_over_k <= 0.05, 1);
k_100_over_100k   = find(eps_100_over_100k <= 0.05, 1);
k_1log_over_k     = find(eps_1log_over_k <= 0.05, 1);
k_1_5log_over_k   = find(eps_1_5log_over_k <= 0.05, 1);

% Plot the curves
figure; hold on;
plot(k, eps_1_over_k,        'b-', 'LineWidth', 1.5);
plot(k, eps_100_over_100k,   'r-', 'LineWidth', 1.5);
plot(k, eps_1log_over_k,     'g-', 'LineWidth', 1.5);
plot(k, eps_1_5log_over_k,   'm-', 'LineWidth', 1.5);

% Add markers for epsilon = 0.05
plot(k_1_over_k, eps_1_over_k(k_1_over_k), 'bo', 'MarkerSize', 7, 'LineWidth', 2);
plot(k_100_over_100k, eps_100_over_100k(k_100_over_100k), 'ro', 'MarkerSize', 7, 'LineWidth', 2);
plot(k_1log_over_k, eps_1log_over_k(k_1log_over_k), 'go', 'MarkerSize', 7, 'LineWidth', 2);
plot(k_1_5log_over_k, eps_1_5log_over_k(k_1_5log_over_k), 'mo', 'MarkerSize', 7, 'LineWidth', 2);

% Label the markers
text(k_1_over_k-10, eps_1_over_k(k_1_over_k)+0.1, sprintf('1/k: k=%d', k_1_over_k));
text(k_100_over_100k -300, eps_100_over_100k(k_100_over_100k)+0.1, sprintf('100/(100+k): k=%d', k_100_over_100k));
text(k_1log_over_k + 30, eps_1log_over_k(k_1log_over_k)+0.2, sprintf('(1+log(k))/k: k=%d', k_1log_over_k));
text(k_1_5log_over_k, eps_1_5log_over_k(k_1_5log_over_k)+0.1, sprintf('(1+5log(k))/k: k=%d', k_1_5log_over_k));

% Final plot formatting
xlabel('k (Iteration Step)');
ylabel('\epsilon (Exploration Rate)');
title('Epsilon Decay Curves from Custom Expressions');
legend('1/k', '100/(100+k)', '(1+log(k))/k', '(1+5log(k))/k', 'Location', 'northeast');
grid on;

%% other epsilon decay methods
% Visualize conservative epsilon decay strategies and mark epsilon = 0.05
clear; clc;

k = 1:10000;

% Define epsilon decay expressions
eps_types = {
    '(1+10log/k)/k', @(k) (1 + 10 * log(k)) ./ k;
    'sqrt(1/k)', @(k) sqrt(1 ./ k);
    'log/k^0.75', @(k) log(k + 1) ./ (k .^ 0.75);
    '0.9^sqrt(k)', @(k) 0.9 .^ sqrt(k);
    '200/(200+k)', @(k) 200 ./ (200 + k);
    '300/(300+k)', @(k) 300 ./ (300 + k);
    '500/(500+k)', @(k) 500 ./ (500 + k);
};

% Prepare figure
figure; hold on;

colors = lines(size(eps_types,1));
os = 0.2;
for i = 1:size(eps_types, 1)
    label = eps_types{i, 1};
    eps_fn = eps_types{i, 2};

    eps_values = eps_fn(k);
    k_cross = find(eps_values <= 0.05, 1);

    plot(k, eps_values, 'LineWidth', 1.5, 'DisplayName', label, 'Color', colors(i,:));

    if ~isempty(k_cross)
        plot(k_cross, eps_values(k_cross), 'o', ...
            'MarkerSize', 7, 'LineWidth', 2, 'Color', colors(i,:));
        text(k_cross -100, eps_values(k_cross)+os, ...
            sprintf('%s: k=%d', label, k_cross), ...
            'Color', colors(i,:), 'FontSize', 9);
    end
    os = os+ 0.15;
end

xlabel('k (Iteration Step)');
ylabel('\epsilon (Exploration Rate)');
title('Conservative Epsilon Decay Strategies');
legend('Location', 'northeast');
grid on;