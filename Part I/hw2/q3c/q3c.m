clc;
clear;
close all;

% Load dataset (ensure variable names match data.mat structure)
load('../data.mat'); 

%% Data Preparation for PCA Analysis
% Merge all samples for PCA evaluation (analysis without dimensionality reduction)
all_images = [X_train, X_test]; % 1024×1000 matrix
retain_components = PCA_dim(all_images); % Determine optimal PCA dimensions

%% Label Format Conversion
% Convert binary labels to class indices (0/1 → 1/2)
Y_train_idx = double(Y_train) + 1;    % Logical → numeric conversion
Y_test_idx = double(Y_test) + 1;      % Logical → numeric conversion
Y_onehot = ind2vec([Y_train_idx, Y_test_idx]); % Create one-hot encoded matrix

%% Data Merging
X_all = [X_train, X_test]; % Preserve original 1024D feature space

%% Network Configuration
trainFcn = 'trainscg';  % Scaled Conjugate Gradient backpropagation

% Initialize MLP with PCA-determined architecture
hiddenLayerSize = retain_components; % 139 units from PCA analysis
setdemorandstream(4912183); % Fix random seed for reproducibility

% Create pattern recognition network
net = patternnet(hiddenLayerSize, trainFcn);

%% Training Parameter Specification
% net.trainParam.show = 1;
% net.trainParam.showCommandLine = true;
net.trainParam.showWindow = true;    % Enable training GUI
net.trainParam.lr = 0.01;            % Learning rate
net.trainParam.epochs = 200;         % Maximum training epochs
net.trainParam.goal = 1e-6;          % Performance threshold
net.trainParam.max_fail = 100;       % Early stopping patience
net.trainParam.min_grad = 1e-8;      % Gradient magnitude threshold

%% Custom Data Partitioning
net.divideFcn = 'divideind';         % Manual dataset splitting
net.divideParam.trainInd = 1:800;    % Training set indices
net.divideParam.valInd = 801:900;    % Validation set indices
net.divideParam.testInd = 901:1000;  % Test set indices

%% Model Training
x = X_all; % Full dataset
t = Y_onehot; % Target matrix
[net, tr] = train(net, x, t); % Train network using original high-dimensional data

% Save model artifacts
save(sprintf('mlp_net_%s.mat', trainFcn), 'net'); 
save(sprintf('mlp_tr_%s.mat', trainFcn), 'tr');

%% Model Evaluation
y = net(x); % Network predictions
% e = gsubtract(t,y);
% performance = perform(net,t,y);
% tind = vec2ind(t);
% yind = vec2ind(y);
% percentErrors = sum(tind ~= yind)/numel(tind);

%% Visualization
fig1 = figure; plotperform(tr); % Training performance metrics
saveas(fig1,sprintf('MLP_perf_%s.png', trainFcn));
fig2 = figure; plottrainstate(tr); % Training parameter dynamics
saveas(fig2,sprintf('MLP_trainstate_%s.png', trainFcn));

%% Accuracy Calculation
train_pred = vec2ind(net(X_train)); % Training set predictions
train_acc = sum(train_pred == Y_train_idx)/900; % 900 training samples
test_pred = vec2ind(net(X_test)); % Test set predictions
test_acc = sum(test_pred == Y_test_idx)/100; % 100 test samples

% Display results
fprintf('Training Accuracy: %.2f%%\n', train_acc*100);
fprintf('Test Accuracy: %.2f%%\n', test_acc*100);




% clc
% clear
% close all
% 
% % 加载数据（确保变量名与data.mat一致）
% load('../data.mat'); 
% 
% % 合并所有数据用于PCA（仅分析，不降维）
% all_images = [X_train, X_test]; % 1024×1000
% retain_components = PCA_dim(all_images);
% 
% % 转换标签格式（0/1 → 1/2类别索引）
% Y_train_idx = double(Y_train) + 1;    % logical → double, 0/1 → 1/2
% Y_test_idx = double(Y_test) + 1;      % logical → double, 0/1 → 1/2
% Y_onehot = ind2vec([Y_train_idx, Y_test_idx]); % 合并为one-hot编码
% 
% % 合并原始数据（保持原始维度）
% X_all = [X_train, X_test]; % 1024×1000
% 
% % Choose a Training Function
% trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
% 
% % Create a Pattern Recognition Network
% hiddenLayerSize = retain_components; %139
% setdemorandstream(4912183); % 固定随机种子，为了复现结果
% net = patternnet(hiddenLayerSize, trainFcn);
% 
% % 配置参数显示选项
% net.trainParam.showWindow = true;
% % net.trainParam.show = 1;
% % net.trainParam.showCommandLine = true;
% 
% % 训练参数配置
% net.trainparam.lr=0.01;
% net.trainParam.epochs = 200;
% net.trainparam.goal=1e-6;       % performance goal
% net.trainParam.max_fail = 100;   % 允许更多次验证失败
% net.trainParam.min_grad = 1e-8; % 梯度阈值
% 
% % 配置数据划分
% net.divideFcn = 'divideind';
% net.divideParam.trainInd = 1:800;    % 训练集索引
% net.divideParam.valInd = 801:900;    % 验证集索引
% net.divideParam.testInd = 901:1000;  % 测试集索引
% 
% % 训练（使用原始数据）
% x = X_all;
% t = Y_onehot;
% [net, tr] = train(net, x, t);
% save(sprintf('mlp_net_%s.mat', trainFcn), 'net'); % Save trained network
% save(sprintf('mlp_tr_%s.mat', trainFcn), 'tr');   % Save training record
% 
% % Test the Network
% y = net(x);
% % e = gsubtract(t,y);
% % performance = perform(net,t,y);
% % tind = vec2ind(t);
% % yind = vec2ind(y);
% % percentErrors = sum(tind ~= yind)/numel(tind)
% 
% % Plots
% fig1=figure; plotperform(tr);
% saveas(fig1,sprintf('MLP_perf_%s.png', trainFcn));
% fig2=figure; plottrainstate(tr);
% saveas(fig2,sprintf('MLP_trainstate_%s.png', trainFcn));
% 
% % Accuracy
% train_pred = vec2ind(net(X_train));
% train_acc = sum(train_pred == Y_train_idx) / 900;
% test_pred = vec2ind(net(X_test));
% test_acc = sum(test_pred == Y_test_idx) / 100;
% fprintf('accuracy_train: %.02f%%\n',train_acc*100);
% fprintf('accuracy_test: %.02f%%\n',test_acc*100);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %% 获取警告标识符
% % [~, warnId] = lastwarn; % 获取最后一个警告的标识符
% % disp(warnId); % 显示标识符（例如 "MATLAB:nnMex:ConvertSparseToFull"）
% 
% 
% 
