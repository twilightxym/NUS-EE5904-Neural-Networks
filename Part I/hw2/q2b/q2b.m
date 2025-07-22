clc
clear
close all

%% Mode selector
mode = 1; % 0:Sequential, 1:Batch

if mode==1
%% Batch mode training using fitnet
% Generate training data (step=0.05)
x_train = -1.6:0.05:1.6;
y_train = 1.2 * sin(pi * x_train) - cos(2.4 * pi * x_train);

% Generate test data (step=0.01)
x_test = -1.6:0.01:1.6;
y_test = 1.2 * sin(pi * x_test) - cos(2.4 * pi * x_test);

% Extended domain for visualization
x_extended = -3:0.01:3; 
y_extended_true = 1.2 * sin(pi * x_extended) - cos(2.4 * pi * x_extended);

% Extrapolation test points
x_extrapolate = [-3,3];

% Network architectures to test
hidden_neurons = [1:10, 20, 50, 100];

% Initialize results structure
results = struct('n', [], 'net', [], 'mse_train', [], 'mse_test', [],...
                 'y_pred',[],'y_extended_pred', [], 'y_extrapolate', []);

% Architecture comparison loop
for i = 1:length(hidden_neurons)
    n = hidden_neurons(i);

    % Network configuration
    net = fitnet(n);                     % Create 1-n-1 network
    net.trainFcn = 'trainlm';            % Levenberg-Marquardt algorithm
    net.trainParam.epochs = 1000;        % Max epoches
    net.trainParam.goal = 1e-5;          % Training target
    net.divideFcn = 'dividetrain';       % No data division
    net.divideParam.trainRatio = 1;      % 100% training data
    net.divideParam.valRatio = 0;        % No validation set
    net.divideParam.testRatio = 0;       % No test set
    net.trainParam.showWindow = false;  % Disable training GUI

    % Network training
    [net, tr] = train(net, x_train, y_train);

    % Model evaluation
    y_pred = net(x_test);                % Test set prediction
    y_extended_pred = net(x_extended);   % Extended domain prediction
    y_extrapolate = net(x_extrapolate);  % Extrapolation points

    % Store results
    results(i).n = n;
    results(i).net = net;
    results(i).mse_train = perform(net, y_train, net(x_train)); % Training MSE
    results(i).mse_test = perform(net, y_test, y_pred);         % Test MSE
    results(i).y_pred = y_pred;
    results(i).y_extended_pred = y_extended_pred;
    results(i).y_extrapolate = y_extrapolate;
end

%% Visualization
% Fitting results visualization
fig1 = figure('WindowState', 'maximized');
for i = 1:length(hidden_neurons)
    subplot(4, 4, i);
    % Plot ground truth vs predictions
    plot(x_extended, y_extended_true, 'b-', 'LineWidth', 1.5); hold on;
    plot(x_extended, results(i).y_extended_pred, 'r--', 'LineWidth', 1);
    % Mark training domain boundaries
    xline(-1.6,'--', 'Training Domain','LineWidth',1); 
    xline(1.6,'--','LineWidth',1);
    % Figure formatting
    title(sprintf('Hidden Neurons = %d', hidden_neurons(i)));
    xlabel('Input x'); 
    ylabel('Output y'); 
    legend('Ground Truth', 'Prediction');
    grid on;
    ylim([-3, 3]); % Unified y-axis limits
end
saveas(fig1,'BatchMode_FittingResults_lm.png');

% Error curve analysis
mse_train = [results.mse_train];
mse_test = [results.mse_test];

fig2 = figure;
semilogy(hidden_neurons, mse_train, 'bo-', 'LineWidth', 1.5); hold on;
semilogy(hidden_neurons, mse_test, 'rs--', 'LineWidth', 1.5);
xlabel('Number of Hidden Neurons');
ylabel('MSE (log scale)');
legend('Training Error', 'Test Error');
title('Model Complexity vs Generalization Error');
grid on;
saveas(fig2,'BatchMode_ErrorCurves_lm.png');

% Extrapolation results
y_extrapolate_true = 1.2 * sin(pi .* x_extrapolate) - cos(2.4 * pi .* x_extrapolate);
fprintf('True Values: y(-3)= %7.3f, y(+3) = %7.3f\n',...
    y_extrapolate_true(1), y_extrapolate_true(2));
fprintf('\nExtrapolation Predictions:\n');
for i = 1:length(hidden_neurons)
    fprintf('n = %2d: y(-3)= %7.3f, y(+3)= %7.3f\n', ...
        hidden_neurons(i), results(i).y_extrapolate(1), results(i).y_extrapolate(2));
end

end
