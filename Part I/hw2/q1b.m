clc
clear
close all

eta = 0.001; % learning rate
% eta = 1.0;
w = [rand(); rand()]; % random initialization of the start point
max_iter = 1e5;
tol = 1e-8;
trajectory = zeros(2, max_iter);
f_values = zeros(1, max_iter);

for iter = 1:max_iter
    x = w(1); y = w(2);
    grad = [-2*(1 - x) - 400*x*(y - x^2); 200*(y - x^2)];
    w_new = w - eta * grad;

    trajectory(:, iter) = w;
    f_values(iter) = (1 - x)^2 + 100*(y - x^2)^2;

    if norm(w_new - w) < tol
    % if f_values(iter) < tol % f(x, y) close to the target value (0)
        break;
    end
    w = w_new;
end

% Plot
fig1 = figure;
plot(trajectory(1, 1:iter), trajectory(2, 1:iter), 'ro-');
hold on;
plot(1, 1, 'c*'); % (1,1) is the global minimum point
title(sprintf('Trajectory of Gradient Descent (eta=%.3f, iter=%d)',eta, iter));
xlabel('x'); ylabel('y');
hold off
saveas(fig1,'Traj_eta0.001.png');

fig2 = figure;
semilogy(f_values(1:iter),'go-');
title(sprintf('Function Value Convergence (eta=%.3f, iter=%d)',eta, iter));
xlabel('Iteration'); ylabel('f(x,y)');
saveas(fig2,'Func_value_eta0.001.png');

%% Plot the Rosenbrock's Valley
f = @(x,y) (1 - x).^2 + 100*(y - x.^2).^2;
[x,y] = meshgrid(linspace(-2, 2, 400),linspace(-1, 3, 400));
z = f(x, y);
[dx, dy] = gradient(z);         % Gradient
grad_mag = sqrt(dx.^2 + dy.^2); % Magnitude of gradient

fig3 = figure;
s = surf(x, y, z, grad_mag, 'EdgeColor', 'none');
colormap(turbo);
clim([min(grad_mag(:)), max(grad_mag(:))]);
colorbar;
title('Rosenbrock Valley Colored by Gradient Magnitude: Blue (Flat) -> Red (Steep)');
xlabel('x');
ylabel('y');
zlabel('f(x,y)');
view(-45, 45);
grid on;
saveas(fig3,'Rosenbrock''s valley.png');


