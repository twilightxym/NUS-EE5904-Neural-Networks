function y_pred = predict(X_train, y_train, X_test, alpha, b, kernel_type, varargin)
    % discriminant function g(x)
    K_test = kernel(X_train, X_test, kernel_type, varargin{:});
    g = sum((alpha .* y_train) .* K_test, 1)' + b;
    y_pred = sign(g);
end