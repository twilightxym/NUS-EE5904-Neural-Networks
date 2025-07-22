clc
clear
close all

% load dataset
img_path= '../group_1';
[X,Y,X_train, Y_train, X_test, Y_test]=load_dataset(img_path);

% save as Data.mat
save('Data.mat', 'X_train', 'Y_train', 'X_test', 'Y_test');

% train perceptron 'trains'
[net, tr] = perceptron_train(X_train, Y_train, X_test, Y_test);




