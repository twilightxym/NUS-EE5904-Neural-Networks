%% Visualize policy as triangles
function drawOptPol(policy, filename)
    figure; hold on;
    title('Optimal Policy');
    for i = 1:11
        xline(i-0.5, 'Color', [0.8 0.8 0.8]);
        yline(i-0.5, 'Color', [0.8 0.8 0.8]);
    end
    for s = 1:100
        row = mod(s - 1, 10) + 1;
        col = ceil(s / 10);
        cx = col; cy = row;
        a = policy(s);
        switch a
            case 1, dx = [0 -0.2 0.2]; dy = [-0.25 0.15 0.15]; % up
            case 2, dx = [-0.15 -0.15 0.25]; dy = [-0.2 0.2 0];% right
            case 3, dx = [0 -0.2 0.2]; dy = [0.25 -0.15 -0.15];% down
            case 4, dx = [0.25 -0.15 -0.15]; dy = [0 -0.2 0.2];% left
        end
        fill(cx + dx, cy + dy, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r');
    end
    xticks(1:10); yticks(1:10);
    axis equal;
    set(gca, 'YDir', 'reverse');
    xlim([0.5 10.5]); ylim([0.5 10.5]);
    saveas(gcf, filename);
end