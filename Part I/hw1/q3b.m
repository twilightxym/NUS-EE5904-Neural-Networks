clc
clear
close all

AND = [0 0 1 0; 0 1 1 0; 1 0 1 0; 1 1 1 1];
OR = [0 0 1 0; 0 1 1 1; 1 0 1 1; 1 1 1 1];
COMPLEMENT = [0 1 1; 1 1 0];
NAND = [0 0 1 1; 0 1 1 1; 1 0 1 1; 1 1 1 0];

eta = 5;


% AND
w = randn(1, 3);
w_h = w;

classified = false;
while ~classified
    classified = true;
    for i = 1:length(AND(:,1))
        if AND(i,1:3)*w'>=0
            y=1;
        else
            y=0;
        end
        if AND(i,4)~= y
            w = w + eta*(AND(i,4)-y)*AND(i,1:3);
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
title(sprintf('Weight Updates for AND gate (LR = %.2f)', eta));
hold off;
saveas(fig1, sprintf('AND_w_%.3f.png', eta));

% x = linspace(0, 2);
% y = -w(1)/w(2)*x - w(3)/w(2);
% fig1a = figure;
% hold on;
% plot(x, y);
% scatter(AND(:,1),AND(:,2),[],AND(:,4),'filled');
% xlim([0 2])
% ylim([0 2])
% title(sprintf('AND (learning rate = %.2f)',eta));
% hold off
% saveas(fig1a,sprintf('AND_PLOT_%.3f.png',eta))


% OR
w = randn(1, 3);
w_h = w;

classified = false;
while ~classified
    classified = true;
    for i = 1:length(OR(:,1))
        if OR(i,1:3)*w'>=0
            y=1;
        else
            y=0;
        end
        if OR(i,4)~= y
            w = w + eta*(OR(i,4)-y)*OR(i,1:3);
            w_h = [w_h; w];
            classified = false;
        end
    end
end

fig2 = figure;
hold on;
n = length(w_h);
plot(0:n-1, w_h(:,1),'-or');
plot(0:n-1, w_h(:,2),'-og');
plot(0:n-1, w_h(:,3),'-ob');
legend({'w1','w2','b'});
title(sprintf('Weight Updates for OR gate(LR = %.2f)',eta));
hold off;
saveas(fig2, sprintf('OR_w_%.3f.png', eta));

% x = linspace(0, 2);
% y = -w(1)/w(2)*x - w(3)/w(2);
% fig2a = figure;
% hold on;
% plot(x, y);
% scatter(OR(:,1),OR(:,2),[],OR(:,4),'filled');
% xlim([0 2])
% ylim([0 2])
% title(sprintf('OR (learning rate = %.2f)',eta));
% hold off
% saveas(fig2a,sprintf('OR_PLOT_%.3f.png',eta))


% NAND
w = randn(1, 3);
w_h = w;

classified = false;
while ~classified
    classified = true;
    for i = 1:length(NAND(:,1))
        if NAND(i,1:3)*w'>=0
            y=1;
        else
            y=0;
        end
        if NAND(i,4)~= y
            w = w + eta*(NAND(i,4)-y)*NAND(i,1:3);
            w_h = [w_h; w];
            classified = false;
        end
    end
end

fig3 = figure;
hold on;
n = length(w_h);
plot(0:n-1, w_h(:,1),'-or');
plot(0:n-1, w_h(:,2),'-og');
plot(0:n-1, w_h(:,3),'-ob');
legend({'w1','w2','b'});
title(sprintf('Weight Updates for NAND gate(LR = %.2f)',eta));
hold off;
saveas(fig3, sprintf('NAND_w_%.3f.png', eta));


% x = linspace(0, 2);
% y = -w(1)/w(2)*x - w(3)/w(2);
% fig3a = figure;
% hold on;
% plot(x, y);
% scatter(NAND(:,1),NAND(:,2),[],NAND(:,4),'filled');
% xlim([0 2])
% ylim([0 2])
% title(sprintf('NAND (learning rate = %.2f)',eta));
% hold off
% saveas(fig3a,sprintf('NAND_PLOT_%.3f.png',eta))


% COMPLEMENT
w = randn(1,2);
w_h = w;

classified = false;
while ~classified
    classified = true;
    for i = 1:length(COMPLEMENT(:,1))
        if COMPLEMENT(i,1:2)*w'>=0
            y=1;
        else
            y=0;
        end
        if COMPLEMENT(i,3)~= y
            w = w + eta*(COMPLEMENT(i,3)-y)*COMPLEMENT(i,1:2);
            w_h = [w_h; w];
            classified = false;
        end
    end
end

fig4 = figure;
hold on;
n = length(w_h);
plot(0:n-1, w_h(:,1),'-or');
plot(0:n-1, w_h(:,2),'-og');
legend({'w1','b'});
title(sprintf('Weight Updates for COMPLEMENT gate(LR = %.2f)',eta));
hold off;
saveas(fig4, sprintf('COMPLEMENT_w_%.3f.png', eta));


% x = - w(2)/w(1);
% fig4a = figure;
% hold on;
% plot([x x], [0 2]);
% scatter(COMPLEMENT(:,1),zeros([length(COMPLEMENT(:,1)) 1]),[],COMPLEMENT(:,3),'filled');
% xlim([0 2])
% ylim([0 2])
% title(sprintf('COMPLEMENT (learning rate = %.2f)',eta));
% hold off
% saveas(fig4a,sprintf('COMPLEMENT_PLOT_%.3f.png',eta))