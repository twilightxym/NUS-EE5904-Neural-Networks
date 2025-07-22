clc;
clear all;
close all;

load('../Data.mat')
X_all = [X_train,X_test];
Y_all = [Y_train,Y_test];

[ net, accu_train, accu_val ] = train_seq(139, X_all, Y_all, 900, 20);
