clc
clear
close all

load('../data.mat'); 
Y_train_idx = double(Y_train) + 1;
Y_test_idx = double(Y_test) + 1;
Y_onehot = ind2vec([Y_train_idx, Y_test_idx]);
X_all = [X_train, X_test];
trainFcn = 'traingdx';
hiddenLayerSize = 139;
setdemorandstream(4912183);

net = patternnet(hiddenLayerSize, trainFcn);

net.trainParam.showWindow = true;
net.trainparam.lr=0.01;
net.trainParam.epochs = 200;
net.trainparam.goal=1e-6;
net.trainParam.max_fail = 100;
net.trainParam.min_grad = 1e-8;

reg_param = 0.9;% Regularization strength 0,0.5,0.9
net.performParam.regularization =reg_param; 

net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:800;
net.divideParam.valInd = 801:900;
net.divideParam.testInd = 901:1000;

x = X_all;
t = Y_onehot;
[net, tr] = train(net, x, t);
% Test the Network
y = net(x);

% Plots
fig1 = figure; plotperform(tr); % Training performance metrics
saveas(fig1,sprintf('%.2f_MLP_perf_%s.png',reg_param, trainFcn));
fig2 = figure; plottrainstate(tr); % Training parameter dynamics
saveas(fig2,sprintf('%.2f_MLP_trainstate_%s.png',reg_param, trainFcn));

% Accuracy
train_pred = vec2ind(net(X_train));
train_acc = sum(train_pred == Y_train_idx) / 900;
test_pred = vec2ind(net(X_test));
test_acc = sum(test_pred == Y_test_idx) / 100;
fprintf('Regularization strength(%s): %.02f\n',trainFcn, reg_param);
fprintf('Accuracy_train: %.02f%%\n',train_acc*100);
fprintf('Accuracy_test: %.02f%%\n',test_acc*100);
fprintf('Norm of input-hidden weights: %f\n', norm(net.IW{1,1}));
fprintf('Norm of hidden-output weights: %f\n', norm(net.LW{2,1}));