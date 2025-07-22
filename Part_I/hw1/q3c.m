clc
clear
close all

XOR = [0 0 1 0; 0 1 1 1; 1 0 1 1; 1 1 1 0];

eta = 1.0;
% XOR
w = randn(1, 3);
w_h = w;

classified = false;
iteration = 100;
while ~classified & iteration
    classified = true;
    iteration = iteration-1;
    for i = 1:length(XOR(:,1))
        if XOR(i,1:3)*w'>=0
            y=1;
        else
            y=0;
        end
        if XOR(i,4)~= y
            w = w + eta*(XOR(i,4)-y)*XOR(i,1:3);
            w_h = [w_h; w];
            classified = false;
        end
    end
end

fig1 = figure;
hold on;
n = length(w_h);
plot(0:n-1, w_h(:,1),'-or');
plot(0:n-1, w_h(:,2),'-og');
plot(0:n-1, w_h(:,3),'-ob');
legend({'w1','w2','b'});
title(sprintf('Weight Updates for XOR gate (LR = %.2f)', eta));
hold off;
saveas(fig1, sprintf('XOR_w_%.3f.png', eta));

x = linspace(0, 2);
y = -w(1)/w(2)*x - w(3)/w(2);
fig1a = figure;
hold on;
plot(x, y);
scatter(XOR(:,1),XOR(:,2),[],XOR(:,4),'filled');
xlim([0 2])
ylim([0 2])
title(sprintf('XOR (learning rate = %.2f)',eta));
hold off
saveas(fig1a,sprintf('XOR_PLOT_%.3f.png',eta))