function is_valid = validate_mercer_condition(K, tol)
    % check if the Gram matrix is positive semi-definite
    % - tol: tolerant residual for check whether eigenvalues are nonnegative (default: 1e-12）

    if nargin < 2
        tol = 1e-12;
    end
    
    % Assure K is symmetric
    K = 0.5 * (K + K');
    
    % Compute eigenvalues
    eigenvalues = eig(K);
    
    % Order-of-magnitude reference for relative tolerant residual: max eigenvalue
    max_eig = max(abs(eigenvalues));
    
    % % relative tolerant：tol * max_eig
    is_valid = all(eigenvalues >= -tol * max_eig);
end
