% function [net, mse_train] = train_seq(n, x_train, y_train, epochs) 
%     x_train_c = num2cell(x_train);
%     y_train_c = num2cell(y_train);
% 
%     net = fitnet(n,'trainlm'); % Levenberg-Marquardt backpropagation
%     net.divideFcn = 'dividetrain'; % input for training only
%     net.divideParam.trainRatio = 1; % all train
%     net.divideParam.testRatio = 0; % no test
%     net.divideParam.valRatio = 0; % no val
%     net.trainParam.epochs = epochs;
%     net.trainParam.showWindow = false;
% 
%     mse_train = zeros(epochs, 1);
% 
%     for epoch = 1:epochs
%         idx = randperm(length(x_train_c));
%         x_train_shuffled = x_train_c(idx);
%         y_train_shuffled = y_train_c(idx);
%         [net, ~, ~] = adapt(net, x_train_shuffled, y_train_shuffled);
% 
%         pred_train = net(x_train);
%         mse_train(epoch) = mean((pred_train - y_train).^2);
%     end
% end

function [net, mse_train] = train_seq(n, x_train, y_train, epochs) 
% TRAIN_SEQ Train 1-n-1 MLP using sequential mode with Levenberg-Marquardt
% Inputs:
%   n       - Number of hidden neurons
%   x_train - Training input data (row vector)
%   y_train - Training target output (row vector)
%   epochs  - Total training epochs
% Outputs:
%   net       - Trained network object
%   mse_train - Training MSE per epoch (epochs×1 vector)

    % Convert matrix to cell array for adapt() compatibility
    x_train_c = num2cell(x_train);
    y_train_c = num2cell(y_train);

    % Network configuration
    net = fitnet(n,'trainlm');          % 1-n-1 architecture with LM algorithm
    net.divideFcn = 'dividetrain';      % Use all data for training
    net.divideParam.trainRatio = 1;     % 100% training data
    net.divideParam.testRatio = 0;      % No test set
    net.divideParam.valRatio = 0;       % No validation set
    net.trainParam.epochs = epochs;     % Maximum training epochs
    net.trainParam.showWindow = false;  % Disable training GUI

    % Initialize MSE recording
    mse_train = zeros(epochs, 1);

    % Sequential training loop
    for epoch = 1:epochs
        % Shuffle training samples
        idx = randperm(length(x_train_c));
        x_train_shuffled = x_train_c(idx);
        y_train_shuffled = y_train_c(idx);
        
        % Single epoch training (sample-wise updates)
        [net, ~, ~] = adapt(net, x_train_shuffled, y_train_shuffled);

        % Calculate training MSE
        pred_train = net(x_train);
        mse_train(epoch) = perform(net, y_train, pred_train);
    end
end





%% V1.0=======================================================================================
% function [net, mse_train, mse_val] = train_seq(n, x_train, y_train, x_val, y_val, epochs)
% % 回归任务专用的 Sequential Training 函数
% % 输入:
% %   n: 隐藏层神经元数量
% %   x_train: 训练集输入 (1×N 行向量)
% %   y_train: 训练集输出 (1×N 行向量)
% %   x_val: 验证集输入 (1×M 行向量)
% %   y_val: 验证集输出 (1×M 行向量)
% %   epochs: 训练轮数
% % 输出:
% %   net: 训练好的网络
% %   mse_train: 各轮训练集 MSE (epochs×1)
% %   mse_val: 各轮验证集 MSE (epochs×1)
% 
% % 转换为 cell 格式 (adapt 要求)
% x_train_c = num2cell(x_train, 1);
% y_train_c = num2cell(y_train, 1);
% x_val_c = num2cell(x_val, 1);
% y_val_c = num2cell(y_val, 1);
% 
% % 构建网络 (1-n-1 结构)
% net = fitnet(n);
% net.trainFcn = 'trainscg';          % Scaled Conjugate Gradient
% net.trainParam.epochs = 1;          % 每次 adapt 视为 1 epoch
% net.trainParam.showWindow = false;  % 关闭训练窗口
% net.divideFcn = 'dividetrain';      % 所有数据用于训练
% net = configure(net, x_train_c, y_train_c);
% 
% % 初始化记录
% mse_train = zeros(epochs, 1);
% mse_val = zeros(epochs, 1);
% 
% % 手动实现早停机制
% best_mse = Inf;
% patience = 100;%20
% wait = 0;
% 
% % Sequential Training
% for epoch = 1:epochs
%     % 打乱训练数据
%     idx = randperm(length(x_train_c));
%     x_train_shuffled = x_train_c(idx);
%     y_train_shuffled = y_train_c(idx);
% 
%     % 逐样本更新
%     for i = 1:length(x_train_shuffled)
%         [net, ~, ~] = adapt(net, x_train_shuffled(i), y_train_shuffled(i));
%     end
% 
%     % 记录当前 MSE
%     pred_train = net(x_train);
%     mse_train(epoch) = mean((pred_train - y_train).^2);
% 
%     pred_val = net(x_val);
%     mse_val(epoch) = mean((pred_val - y_val).^2);
% 
%     % Early Stopping 逻辑
%     if mse_val(epoch) < best_mse
%         best_mse = mse_val(epoch);
%         best_net = net;
%         wait = 0;
%     else
%         wait = wait + 1;
%         if wait >= patience
%             fprintf('Early stopping at epoch %d\n', epoch);
%             break;
%         end
%     end
% end
% 
% % 恢复最佳网络
% net = best_net;
% end

%=======================================================================================
% function [ net, accu_train, accu_val ] = train_seq( n, x, y, train_num, epochs)
% % Construct a 1-n-1 MLP and conduct sequential training.
% %
% % Args:
% % n: int, number of neurons in the hidden layer of MLP.
% % images: matrix of (image_dim, image_num), containing possibly
% % preprocessed image data as input.
% % labels: vector of (1, image_num), containing corresponding label of each
% % image.
% % train_num: int, number of training images.
% % epochs: int, number of training epochs.
% %
% % Returns:
% % net: object, containing trained network.
% % accu_train: vector of (epochs, 1), containing the accuracy on training
% % set of each eopch during trainig.
% 
% % accu_val: vector of (epochs, 1), containing the accuracy on validation
% % set of each eopch during trainig.
% % 1. Change the input to cell array form for sequential training
% x_c = num2cell(x, 1);
% y_c = num2cell(y, 1);
% 
% % 2. Construct and configure the MLP
% net = fitnet(n);
% 
% net.divideFcn = 'dividerand';
% net.divideParam.trainRatio = 0.7;
% net.divideParam.valRatio = 0.3;
% net.divideParam.testRatio = 0;
% net.trainParam.max_fail = 20; % Early stopping 验证误差连续上升20次后停止
% % net.performParam.regularization = 0.25; % regularization strength
% net.trainFcn = 'trainscg'; % 'trainrp' 'traingdx'  % Here using Scaled conjugate gradient backpropagation
% net.trainParam.epochs = epochs;
% 
% accu_train = zeros(epochs,1); % record accuracy on training set of each epoch
% accu_val = zeros(epochs,1); % record accuracy on validation set of each epoch
% 
% % 3. Train the network in sequential mode
% for i = 1 : epochs
%     display(['Epoch: ', num2str(i)])
%     idx = randperm(train_num); % shuffle the input
%     net = adapt(net, x_c(:,idx), y_c(:,idx));
% 
%     pred_train = net(x(:,1:train_num)); % predictions on training set
%     accu_train(i) = 1 - mean(abs(pred_train-y(1:train_num)));
% 
%     pred_val = net(x(:,train_num+1:end)); % predictions on validation set
%     accu_val(i) = 1 - mean(abs(pred_val-y(train_num+1:end)));
% 
% end
% end

%% V2.0
% function [net, mse_train, mse_val] = train_seq(n, x_train, y_train, x_val, y_val, epochs)
% % 回归任务专用的 Sequential Training 函数（无归一化） V2.0
% % 输入:
% %   n: 隐藏层神经元数量
% %   x_train: 训练集输入 (1×N)
% %   y_train: 训练集输出 (1×N)
% %   x_val: 验证集输入 (1×M)
% %   y_val: 验证集输出 (1×M)
% %   epochs: 最大训练轮数
% % 输出:
% %   net: 训练好的网络
% %   mse_train: 各轮训练集 MSE (epochs×1)
% %   mse_val: 各轮验证集 MSE (epochs×1)
% 
% % 转换为 cell 格式 (adapt 要求)
% x_train_c = num2cell(x_train);
% y_train_c = num2cell(y_train);
% x_val_c = num2cell(x_val);
% y_val_c = num2cell(y_val);
% 
% % 构建网络 (1-n-1 结构)
% net = fitnet(n);
% net.trainFcn = 'trainscg';          % Scaled Conjugate Gradient
% net.trainParam.epochs = 1;          % 每次 adapt 视为 1 epoch
% net.trainParam.showWindow = false;
% net.divideFcn = 'dividetrain';      % 所有数据用于训练
% net = configure(net, x_train_c, y_train_c);
% 
% % 初始化记录
% mse_train = zeros(epochs, 1);
% mse_val = zeros(epochs, 1);
% 
% % 弹性早停参数
% best_mse = Inf;
% patience = 50;  % 增大耐心值避免过早停止
% wait = 0;
% 
% % Sequential Training
% for epoch = 1:epochs
%     % 打乱训练数据
%     idx = randperm(length(x_train_c));
%     x_train_shuffled = x_train_c(idx);
%     y_train_shuffled = y_train_c(idx);
% 
%     % 逐样本更新
%     for i = 1:length(x_train_shuffled)
%         [net, ~, ~] = adapt(net, x_train_shuffled(i), y_train_shuffled(i));
%     end
% 
%     % 记录当前 MSE
%     pred_train = net(x_train);
%     mse_train(epoch) = mean((pred_train - y_train).^2);
% 
%     pred_val = net(x_val);
%     mse_val(epoch) = mean((pred_val - y_val).^2);
% 
%     % 弹性早停逻辑：仅当连续patience次无改善才停止
%     if mse_val(epoch) < best_mse * 0.999  % 允许轻微波动
%         best_mse = mse_val(epoch);
%         best_net = net;
%         wait = 0;
%     else
%         wait = wait + 1;
%         if wait >= patience
%             fprintf('Early stopping at epoch %d\n', epoch);
%             break;
%         end
%     end
% end
% 
% % 截断未使用的记录
% mse_train = mse_train(1:epoch);
% mse_val = mse_val(1:epoch);
% net = best_net;
% end