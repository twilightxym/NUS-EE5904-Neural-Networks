%% Visualize trajectory and total reward
function drawTraj(path, total_reward, filename, reward)
    figure; hold on;
    for i = 1:11
        xline(i - 0.5, 'Color', [0.8 0.8 0.8]);
        yline(i - 0.5, 'Color', [0.8 0.8 0.8]);
    end
    reward_undiscount = 0;
    for i = 1:length(path) - 1
        s = path(i); s_next = path(i + 1);
        row = mod(s - 1, 10) + 1;
        col = ceil(s / 10); cx = col; cy = row;
        row2 = mod(s_next - 1, 10) + 1;
        col2 = ceil(s_next / 10);
        dx = col2 - col; dy = row2 - row;

        if dx == 0 && dy == -1
            a = 1;
        elseif dx == 1 && dy == 0
            a = 2;
        elseif dx == 0 && dy == 1
            a = 3;
        elseif dx == -1 && dy == 0
            a = 4;
        else
            continue;
        end

        switch a
            case 1, dxs = [0 -0.2 0.2]; dys = [-0.25 0.15 0.15];   
            case 2, dxs = [-0.15 -0.15 0.25]; dys = [-0.2 0.2 0];
            case 3, dxs = [0 -0.2 0.2]; dys = [0.25 -0.15 -0.15];
            case 4, dxs = [0.25 -0.15 -0.15]; dys = [0 -0.2 0.2];
        end
        fill(cx + dxs, cy + dys, 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'b');

        tmp_reward = reward(s, a);
        reward_undiscount = reward_undiscount+ tmp_reward;
    end
    plot(10, 10, '*r', 'MarkerSize', 10);

    xticks(1:10); yticks(1:10);
    axis equal;
    set(gca, 'YDir', 'reverse');
    xlim([0.5 10.5]); ylim([0.5 10.5]);
    title(sprintf('Optimal Trajectory (Total discounted reward: %.4f, undiscounted reward: %d)', total_reward, reward_undiscount));
    saveas(gcf, filename);
end