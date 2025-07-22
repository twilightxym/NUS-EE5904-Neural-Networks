%% Main Q-learning function
function [opt_policy, success_runs, avg_time, opt_path, max_reward] = Qlearning_main(reward, gamma, eps_type)
    max_episodes = 3000;
    runs = 10;
    goal_state = 100;
    success_runs = 0;
    total_time = 0;
    max_reward = -inf;
    opt_policy = [];
    opt_path = [];

    threshold = 1e-4;       % Q-table convergence threshold
    stable_required = 300;   % how many consecutive times must be stable

    for run = 1:runs
        Q = zeros(100, 4);
        t0 = tic;
        stable_count = 0;

        for episode = 1:max_episodes
            Q_prev = Q;
            s = 1;
            k = 1;
            alpha = getEps(eps_type, k);
            max_steps=10000;

            while true
                max_steps = max_steps-1;
                a = chooseAction(Q, s, eps_type, k, reward);
                [s_next, valid] = move(s, a);
                if ~valid
                    Q(s,a) = Q(s,a) + alpha * (reward(s,a) + gamma * max(Q(s,:)) - Q(s,a));
                    break;
                end
                Q(s,a) = Q(s,a) + alpha * (reward(s,a) + gamma * max(Q(s_next,:)) - Q(s,a));
                s = s_next;
                k = k + 1;
                alpha = getEps(eps_type, k);
                if s == goal_state || alpha < 0.05 || max_steps<0
                    if max_steps<0
                        disp('Max steps: early stopping.')
                    end
                    break;
                end
            end

            % Check convergence
            if max(abs(Q(:) - Q_prev(:))) < threshold
                stable_count = stable_count + 1;
            else
                stable_count = 0;
            end

            if stable_count >= stable_required
                break;
            end
        end

        etime = toc(t0);
        policy = getPolicy(Q);
        traj = replay(policy);

        if traj(end) == goal_state
            success_runs = success_runs + 1;
            total_time = total_time + etime;
            R = getPathReward(reward, traj,gamma);
            if R > max_reward
                max_reward = R;
                opt_policy = policy;
                opt_path = traj;
            end
        end
    end
    avg_time = total_time / max(success_runs, 1);
end

%% Epsilon schedule
function eps = getEps(type, k)
    switch type
        % from the given table
        case '1/k'                          %20
            eps = 1 / k;
        case '100/(100+k)'                  %1900
            eps = 100 / (100 + k);
        case '(1+logk)/k'                   %115
            eps = (1 + log(k)) / k;         
        case '(1+5logk)/k'                  %671
            eps = (1 + 5 * log(k)) / k;

        % more conservative decay strategies
        case 'sqrt(1/k)'                    %400
            eps = sqrt(1 / k);
        case 'log(k+1)/k^0.75'              %658
            eps = log(k + 1) / k^0.75;
        case '0.9^sqrt(k)'                  %809
            eps = 0.9 ^ sqrt(k);
        case '(1+10logk)/k'                 %1480
            eps = (1 + 10 * log(k)) / k;

        % slower decay
        case '200/(200+k)'                  %3800
            eps = 200 / (200 + k);
        case '300/(300+k)'                  %5700
            eps = 300 / (300 + k);
        case '500/(500+k)'                  %9500
            eps = 500 / (500 + k);
        otherwise
            error('Unknown epsilon decay type: %s', type);
    end
end

%% ε-greedy action selection
function a = chooseAction(Q, s, eps_type, k, reward)
    eps = getEps(eps_type, k);
    valid_actions = find(reward(s, :) > -1);

    if rand < eps
        idx = randi(length(valid_actions));
        a = valid_actions(idx);
    else
        [~, best_idx] = max(Q(s, valid_actions));
        a = valid_actions(best_idx);
    end
end

%% Deterministic grid movement
function [s_next, valid] = move(s, a)
    col = ceil(s / 10);
    row = mod(s - 1, 10) + 1;
    valid = true;
    switch a
        case 1, row = row - 1;
        case 2, col = col + 1;
        case 3, row = row + 1;
        case 4, col = col - 1;
    end
    if row < 1 || row > 10 || col < 1 || col > 10
        valid = false;
        s_next = s;
    else
        s_next = (col - 1) * 10 + row;
    end
end

%% Extract policy
function policy = getPolicy(Q)
    [~, policy] = max(Q, [], 2);
end

%% Replay trajectory
function traj = replay(policy)
    s = 1; traj = s;
    while s ~= 100
        a = policy(s);
        [s_next, valid] = move(s, a);
        if ~valid || ismember(s_next, traj)
            break;
        end
        traj(end+1) = s_next;
        s = s_next;
    end
end

%% Cumulative of rewards for a trajectory
function total = getPathReward(reward, traj, gamma)
    total = 0;
    for i = 1:length(traj)-1
        s = traj(i); s_next = traj(i+1);
        for a = 1:4
            [temp, valid] = move(s, a);
            if valid && temp == s_next
                total = total + (gamma^(i-1)) * reward(s, a);  % discounted reward
                break;
            end
        end
    end
end



