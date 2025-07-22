clc
clear
close all

AND = [0 0 1 0;0 1 1 0;1 0 1 0;1 1 1 1];
OR = [0 0 1 0;0 1 1 1;1 0 1 1;1 1 1 1];
NAND = [0 0 1 1;0 1 1 1;1 0 1 1;1 1 1 0];
COMPLEMENT = [0 1 1;1 1 0];

% AND
x = linspace(0,2);
y = -x+1.5;
fig1 = figure;
hold on;
plot(x, y);
scatter(AND(:,1),AND(:,2),[], AND(:,4), "filled")
xlim([0 2])
ylim([0 2])
title('AND (off-line)')
hold off
saveas(fig1,'AND.png');

% OR
x = linspace(0,2);
y = -x+0.5;
fig2 = figure;
hold on;
plot(x, y);
scatter(OR(:,1),OR(:,2),[], OR(:,4), "filled")
xlim([0 2])
ylim([0 2])
title('OR (off-line)')
hold off
saveas(fig2,'OR.png');

% COMPLEMENT
x = 0.5;
fig3 = figure;
hold on;
plot([x x], [0 2]);
scatter(COMPLEMENT(:,1),zeros([length(COMPLEMENT(:,1)) 1]),[], COMPLEMENT(:,3), "filled")
xlim([0 2])
ylim([0 2])
title('COMPLEMENT (off-line)')
hold off
saveas(fig3,'COMPLEMENT.png');

% NAND
x = linspace(0,2);
y = -x+1.5;
fig4 = figure;
hold on;
plot(x, y);
scatter(NAND(:,1),NAND(:,2),[], NAND(:,4), "filled")
xlim([0 2])
ylim([0 2])
title('NAND (off-line)')
hold off
saveas(fig4,'NAND.png');