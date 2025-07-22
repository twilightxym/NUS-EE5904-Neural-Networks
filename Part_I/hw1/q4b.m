clc
clear
close all

x = [0 1;0.8 1;1.6 1;3 1;4.0 1;5.0 1];
d = [0.5; 1; 4; 5; 6; 8];

w = randn(1,2);
w_h = w;
eta = 0.01;

for epoch = 1:100
    for i = 1:length(x(:,1))
        if x(i,:)*w'==d(i)
        else
            e = d(i)-x(i,:)*w';
            w = w+eta*e*x(i,:);
            w_h = [w_h; w];
        end
    end
end

y = x*w';

fig1 = figure
hold on 
scatter(x(:,1),d,'filled')
plot(x(:,1),y);
title(sprintf('Fitting result (LMS, LR=%.4f)', eta));
hold off
saveas(fig1, sprintf('LMS_%.4f.png',eta));

fig2 = figure
hold on 
plot(0:length(w_h)-1,w_h(:,1),'-or')
plot(0:length(w_h)-1,w_h(:,2),'-ob')
legend({'w1','b'});
title(sprintf('weight update (LR=%.4f)', eta));
hold off
saveas(fig2, sprintf('Weight_LMS_%.4f.png', eta));
