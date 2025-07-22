%% Train Perceptron Classifier
function [net, tr] = perceptron_train(X_train, Y_train, X_test, Y_test)
    % Initialize single-layer perceptron
    net = perceptron;
    net = configure(net, X_train, Y_train);
    
    
    % Phase 1: Sequential weight/bias training
    net.trainFcn = 'trains'; % Sequential weight/bias update rule
    [net, tr] = train(net, X_train, Y_train);
    save('slp_net_trains.mat', 'net'); % Save trained network
    save('slp_tr_trains.mat', 'tr');   % Save training record
    fig1=figure; plotperform(tr);
    saveas(fig1,'Perceptron_trains_perf.png');
    fig1=figure; plotconfusion(Y_test,net(X_test));
    saveas(fig1,'Perceptron_trains_confu.png');
    
    % Phase 2: Fine-tuning with cyclical weight updates
    net.trainFcn = 'trainc'; % Cyclical weight/bias update rule
    [net, tr] = train(net, X_train, Y_train); % Refine model parameters
    save('slp_net_trainc.mat', 'net'); % Save optimized network
    save('slp_tr_trainc.mat', 'tr');   % Save updated training record
    fig2=figure; plotperform(tr);
    saveas(fig2,'Perceptron_trainc_perf.png');
    fig2=figure; plotconfusion(Y_test,net(X_test));
    saveas(fig2,'Perceptron_trainc_confu.png');
    
    % Performance evaluation
    acc_train = 1 - mean(abs(net(X_train) - Y_train)); % Training accuracy
    acc_test = 1 - mean(abs(net(X_test) - Y_test));    % Test accuracy
    acc = [acc_train; acc_test]; % Accuracy vector: [train; test]
    
    fprintf('Perceptron: Train Accuracy=%.2f%%, Test Accuracy=%.2f%%\n',...
            acc_train*100, acc_test*100);
    save('acc_test.mat', 'acc'); % Save accuracy metrics
end
