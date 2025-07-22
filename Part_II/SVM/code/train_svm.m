function [alpha, b] = train_svm(X, y, C, kernel_type, varargin)
    check_mercer = false;
    % optional parameter: check_mercer（unused by default, prepared for custom kernel）
    % for instance:
    % custom_kernel = @(X1, X2) (X1' * X2).^3 + 0.5 * (X1' * X2).^2;
    % [alpha, b] = train_svm(X_train, y_train, 1.0, 'custom', custom_kernel, true);
    
    if nargin >= 6 && islogical(varargin{end})
        check_mercer = varargin{end};
        varargin = varargin(1:end-1);
    end

    N = size(X, 2);
    K = kernel(X, X, kernel_type, varargin{:});
    
    if check_mercer
        is_valid = validate_mercer_condition(K);
        if ~is_valid
            warning('Kernel may violate Mercer condition. Results may be unreliable.');
            fprintf('[Mercer''s Cond.] kernel_type:%s, with p=%d (C=%.3f)\n', kernel_type, varargin{:}, C);
        end
    end

    % Quadratic programming setup
    H = (y * y') .* K;
    f = -ones(N, 1);
    Aeq = y';
    beq = 0;
    lb = zeros(N, 1);
    ub = C * ones(N, 1);
    
    % Solve with quadprog
    % options = optimset('LargeScale','off','MaxIter',1000); % maybe outdated
    % alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);

    options = optimoptions('quadprog', ...
    'Algorithm', 'interior-point-convex', ...
    'Display', 'off', ...
    'MaxIterations', 1000);
    
    [alpha, ~, exitflag] = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);
    if exitflag <= 0
        warning('Optimization did not converge, consider increasing MaxIterations or adjusting the algorithm');
        fprintf('[quadprog] kernel_type:%s, with p=%d (C=%.3f) exitflag=%d\n', kernel_type, varargin{:}, C, exitflag);
    end

    % Selection of support vectors (hard margin: 0 < α; soft margin:0 < α ≤ C)
    epsilon = 1e-4;
    if(C>1000)
        sv_idx = alpha > epsilon;
    else
        sv_idx = (alpha > epsilon) & (alpha <= C);
    end
    
    if sum(sv_idx) == 0
        warning('No strict support vectors found, using all α>0');
        sv_idx = alpha > epsilon;
    end
    
    % Compute bias b
    alpha_sv = alpha(sv_idx);
    y_sv = y(sv_idx);
    K_sv = kernel(X, X(:, sv_idx), kernel_type, varargin{:});
    g_sv = sum((alpha_sv .* y_sv) .* K_sv(sv_idx, :), 1)';
    b = mean(y_sv - g_sv);
end