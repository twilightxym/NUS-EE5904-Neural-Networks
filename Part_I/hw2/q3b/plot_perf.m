close all;
load_perf_plot('slp_tr_trains.mat');
saveas(gcf, 'Perceptron_trains_perf.png');

close all;
load_perf_plot('slp_tr_trainc.mat');
saveas(gcf, ['Perceptron_trainc_perf.png']);







%% 用于加载已保存到mat文件的训练记录 tr
function load_perf_plot(path)
% 加载保存的模型
load(path, 'tr'); % 显式加载net和tr变量

% 检查是否存在tr
if exist('tr')
    disp('训练记录tr存在');
else
    disp('警告：未找到训练记录tr，无法绘制性能曲线');
end

plotperf(tr);
xlabel('Epoch');
ylabel('Mean Absolute Error (mae)');
legend('Training');
grid on;
end


