clc
clear
close all

x = [0 1;0.8 1;1.6 1;3 1;4.0 1;5.0 1];
d = [0.5; 1; 4; 5; 6; 8];

w = (x'*x)^(-1)*x'*d; %standard linear least square
y = x*w;

fig = figure
hold on 
scatter(x(:,1),d,'filled')
plot(x(:,1),y);
title('Fitting result (LLS)')
hold off
saveas(fig, 'LLS.png')