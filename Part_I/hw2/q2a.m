clc
clear
close all

mode = 0; % 0:Sequential, 1:Batch

if mode == 0
%% Sequential mode training using train_seq
% Generate raw data
x_train = -1.6:0.05:1.6;
y_train = 1.2 * sin(pi * x_train) - cos(2.4 * pi * x_train);

% Test set for error calculation
x_test = -1.6:0.01:1.6;
y_test = 1.2 * sin(pi * x_test) - cos(2.4 * pi * x_test);

% Extended domain visualization
x_extended = -3:0.01:3; 
y_extended_true = 1.2 * sin(pi * x_extended) - cos(2.4 * pi * x_extended);

% Extrapolation test points
x_extrapolate = [-3,3];

% Network architectures
hidden_neurons = [1:10, 20, 50, 100];
epochs = 1000;  % Maximum training epochs

% Initialize results structure
results = struct('n', [], 'net', [], 'mse_train', [], 'mse_test', [],...
                 'y_pred',[],'y_extended_pred', [], 'y_extrapolate', []);

% Architecture comparison loop
for i = 1:length(hidden_neurons)
    n = hidden_neurons(i);
    fprintf('Training n = %d...\n', n);

    % Network training
    [net, mse_train] = train_seq(n, x_train, y_train, epochs);

    % Model evaluation
    y_pred = net(x_test);                % Test set prediction
    % mse_test = mean((y_pred - y_test).^2); % Test MSE
    y_extended_pred = net(x_extended);   % Extended domain prediction
    y_extrapolate = net(x_extrapolate);  % Extrapolation points

    % Store results
    results(i).n = n;
    results(i).net = net;
    results(i).mse_train = mse_train;
    results(i).mse_test = perform(net, y_test, y_pred);
    results(i).y_pred = y_pred;
    results(i).y_extended_pred = y_extended_pred;
    results(i).y_extrapolate = y_extrapolate;
end

%% Visualization
% Fitting results
fig1 = figure('WindowState', 'maximized');
for i = 1:length(hidden_neurons)
    subplot(4, 4, i);
    % Plot ground truth vs predictions
    plot(x_extended, y_extended_true, 'b-', 'LineWidth', 1.5); hold on;
    plot(x_extended, results(i).y_extended_pred, 'r--', 'LineWidth', 1);
    % Mark training domain boundaries
    xline(-1.6, '--', 'Training Domain','LineWidth',1); 
    xline(1.6, '--','LineWidth',1);
    % Figure formatting
    title(sprintf('Hidden Neurons = %d', hidden_neurons(i)));
    xlabel('Input x'); 
    ylabel('Output y'); 
    legend('Ground Truth', 'Prediction');
    grid on;
    ylim([-3, 3]); % Unified y-axis limits
end
% saveas(fig1,'SequentialMode_FittingResults_1000epoch.png');

% Error analysis
mse_train_final = arrayfun(@(s) s.mse_train(end), results); % Final training MSE
mse_test = arrayfun(@(s) s.mse_test, results);              % Test MSE

fig2 = figure;
semilogy(hidden_neurons, mse_train_final, 'bo-', 'LineWidth', 1.5); hold on;
semilogy(hidden_neurons, mse_test, 'rs--', 'LineWidth', 1.5);
xlabel('Number of Hidden Neurons');
ylabel('MSE (log scale)');
legend('Training Error', 'Test Error');
title('Model Complexity vs Generalization Error');
grid on;
saveas(fig2,'SequentialMode_ErrorCurves_1000epoch.png');

% Extrapolation results
y_extrapolate_true = 1.2 * sin(pi .* x_extrapolate) - cos(2.4 * pi .* x_extrapolate);
fprintf('True Values: y(-3)= %7.3f, y(+3) = %7.3f\n',...
    y_extrapolate_true(1), y_extrapolate_true(2));
fprintf('\nExtrapolation Predictions:\n');
for i = 1:length(hidden_neurons)
    fprintf('n = %2d: y(-3)= %7.3f, y(+3)= %7.3f\n', ...
        results(i).n, results(i).y_extrapolate(1), results(i).y_extrapolate(2));
end

end








%% 生成数据  V1.0
% % 训练集 (步长 0.05)
% x_train = -1.6:0.05:1.6;
% y_train = 1.2 * sin(pi * x_train) - cos(2.4 * pi * x_train);
% 
% % 测试集 (步长 0.01)
% x_test = -1.6:0.01:1.6;
% y_test = 1.2 * sin(pi * x_test) - cos(2.4 * pi * x_test);
% 
% % 归一化 (Min-Max 归一化到 [-1, 1])
% [x_train_norm, x_settings] = mapminmax(x_train, -1, 1);
% [y_train_norm, y_settings] = mapminmax(y_train, -1, 1);
% x_test_norm = mapminmax('apply', x_test, x_settings);
% y_test_norm = mapminmax('apply', y_test, y_settings);
% 
% % 划分训练集和验证集 (70% 训练, 30% 验证)
% train_ratio = 0.7;
% train_num = round(train_ratio * length(x_train_norm));
% x_val_norm = x_train_norm(train_num+1:end);
% y_val_norm = y_train_norm(train_num+1:end);
% x_train_norm = x_train_norm(1:train_num);
% y_train_norm = y_train_norm(1:train_num);
% 
% %% 训练不同结构的网络
% hidden_neurons = [1:10, 20, 50, 100];
% epochs = 500;
% results = struct('n', [], 'net', [], 'mse_train', [], 'mse_val', [], 'y_pred', [], 'y_extrapolate', []);
% 
% for i = 1:length(hidden_neurons)
%     n = hidden_neurons(i);
%     fprintf('Training n = %d...\n', n);
% 
%     % 训练网络
%     [net, mse_train, mse_val] = train_seq(n, x_train_norm, y_train_norm, x_val_norm, y_val_norm, epochs);
% 
%     % 预测测试集
%     y_pred_norm = net(x_test_norm);
%     y_pred = mapminmax('reverse', y_pred_norm, y_settings);
% 
%     % 外推预测 (x=-3 和 x=3)
%     x_extrapolate = [-3, 3];
%     x_extrapolate_norm = mapminmax('apply', x_extrapolate, x_settings);
%     y_extrapolate_norm = net(x_extrapolate_norm);
%     y_extrapolate = mapminmax('reverse', y_extrapolate_norm, y_settings);
% 
%     % 保存结果
%     results(i).n = n;
%     results(i).net = net;
%     results(i).mse_train = mse_train;
%     results(i).mse_val = mse_val;
%     results(i).y_pred = y_pred;
%     results(i).y_extrapolate = y_extrapolate;
% end
% 
% %% 可视化拟合结果
% figure('Position', [100, 100, 1200, 800]);
% for i = 1:length(hidden_neurons)
%     subplot(4, 4, i);
%     plot(x_test, y_test, 'b-', 'LineWidth', 1.5); hold on;
%     plot(x_test, results(i).y_pred, 'r--', 'LineWidth', 1);
%     title(sprintf('n = %d', hidden_neurons(i)));
%     xlabel('x');
%     ylabel('y');
%     legend('True', 'Predicted', 'Location', 'best');
%     grid on;
%     ylim([-3, 3]);
% end
% 
% %% 绘制误差曲线
% mse_train_final = arrayfun(@(s) s.mse_train(end), results);
% mse_val_final = arrayfun(@(s) s.mse_val(end), results);
% 
% figure;
% semilogy(hidden_neurons, mse_train_final, 'bo-', 'LineWidth', 1.5); hold on;
% semilogy(hidden_neurons, mse_val_final, 'rs--', 'LineWidth', 1.5);
% xlabel('Number of Hidden Neurons');
% ylabel('MSE (log scale)');
% legend('Training MSE', 'Validation MSE');
% title('Final MSE vs. Hidden Neurons');
% grid on;
% 
% %% 外推预测结果
% fprintf('\n外推预测结果:\n');
% for i = 1:length(hidden_neurons)
%     fprintf('n = %2d: y(-3) = %8.4f, y(+3) = %8.4f\n', ...
%         results(i).n, results(i).y_extrapolate(1), results(i).y_extrapolate(2));
% end
%-------------------------------------------------------

%% Use train_seq to train MLP V2.0
% %% 生成原始数据（无归一化）
% x_train = -1.6:0.05:1.6;
% y_train = 1.2 * sin(pi * x_train) - cos(2.4 * pi * x_train);
% x_test = -1.6:0.01:1.6;
% y_test = 1.2 * sin(pi * x_test) - cos(2.4 * pi * x_test);
% 
% % 划分训练集和验证集 (70% 训练, 30% 验证)
% train_ratio = 0.7;
% train_num = round(train_ratio * length(x_train));
% x_val = x_train(train_num+1:end);
% y_val = y_train(train_num+1:end);
% x_train = x_train(1:train_num);
% y_train = y_train(1:train_num);
% 
% %% 训练不同结构的网络
% hidden_neurons = [1:10, 20, 50, 100];
% epochs = 1000;  % 增大最大 epoch 数
% results = struct('n', [], 'net', [], 'mse_train', [], 'mse_val', [], 'y_pred', [], 'y_extrapolate', []);
% 
% for i = 1:length(hidden_neurons)
%     n = hidden_neurons(i);
%     fprintf('Training n = %d...\n', n);
% 
%     % 训练网络（直接使用原始数据）
%     [net, mse_train, mse_val] = train_seq(n, x_train, y_train, x_val, y_val, epochs);
% 
%     % 预测测试集
%     y_pred = net(x_test);
% 
%     % 外推预测 (x=-3 和 x=3)
%     x_extrapolate = [-3, 3];
%     y_extrapolate = net(x_extrapolate);
% 
%     % 保存结果
%     results(i).n = n;
%     results(i).net = net;
%     results(i).mse_train = mse_train;
%     results(i).mse_val = mse_val;
%     results(i).y_pred = y_pred;
%     results(i).y_extrapolate = y_extrapolate;
% end
% 
% %% 可视化拟合结果（突出欠拟合/过拟合）
% figure('Position', [100, 100, 1200, 800]);
% for i = 1:length(hidden_neurons)
%     subplot(4, 4, i);
%     plot(x_test, y_test, 'b-', 'LineWidth', 1.5); hold on;
%     plot(x_test, results(i).y_pred, 'r--', 'LineWidth', 1);
%     title(sprintf('n = %d', hidden_neurons(i)));
%     xlabel('x'); ylabel('y'); ylim([-3, 3]);
%     legend('True', 'Predicted', 'Location', 'best');
%     grid on;
% end
% 
% %% 绘制误差分析曲线（训练 vs 验证）
% mse_train_final = arrayfun(@(s) s.mse_train(end), results);
% mse_val_final = arrayfun(@(s) s.mse_val(end), results);
% 
% figure;
% semilogy(hidden_neurons, mse_train_final, 'bo-', 'LineWidth', 1.5); hold on;
% semilogy(hidden_neurons, mse_val_final, 'rs--', 'LineWidth', 1.5);
% xlabel('Number of Hidden Neurons');
% ylabel('Final MSE (log scale)');
% legend('Training', 'Validation');
% title('Model Complexity vs Generalization Error');
% grid on;
% 
% %% 外推预测分析
% fprintf('\n外推预测结果:\n');
% 
% for i = 1:length(hidden_neurons)
%     fprintf('n = %2d: y(-3) = %7.3f, y(+3) = %7.3f\n', ...
%         results(i).n, results(i).y_extrapolate(1), results(i).y_extrapolate(2));
% end
% y_extrapolate_true = 1.2 * sin(pi .* x_extrapolate) - cos(2.4 * pi .* x_extrapolate);
% fprintf('True value: y(-3)= %7.3f, y(+3) = %7.3f\n',...
%     y_extrapolate_true(1), y_extrapolate_true(2));





