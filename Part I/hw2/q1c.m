clc
clear
close all

eta = 0.001; % learning rate
w = [rand(); rand()];
max_iter = 100;
tol = 1e-8;
trajectory = zeros(2, max_iter);
f_values = zeros(1, max_iter);

for iter = 1:max_iter
    x = w(1); y = w(2);
    grad = [-2*(1 - x) - 400*x*(y - x^2); 200*(y - x^2)];
    H = [2 + 400*(3*x^2-y), -400*x; 
         -400*x, 200]; % Hessian Matrix
    delta_w = -inv(H) * grad;
    w_new = w + delta_w;

    trajectory(:, iter) = w;
    f_values(iter) = (1 - x)^2 + 100*(y - x^2)^2;

    if norm(delta_w) < tol
    % if f_values(iter) < tol
        break;
    end
    w = w_new;
end

% Plot
fig1=figure;
plot(trajectory(1, 1:iter), trajectory(2, 1:iter), 'ro-');
hold on;
plot(1, 1, 'c*');  %(1,1) is the global minimum point
title(sprintf('Trajectory of Gradient Descent (eta=%.3f, iter=%d)',eta, iter));
xlabel('x'); ylabel('y');
hold off;
saveas(fig1,'Traj_eta0.001.png');

fig2=figure;
semilogy(f_values(1:iter),'go-');
title(sprintf('Function Value Convergence (eta=%.3f, iter=%d)',eta, iter));
xlabel('Iteration'); ylabel('f(x,y)');
saveas(fig2,'Func_value_eta0.001.png');


% 大学习率测试（η=1.0时发散）
