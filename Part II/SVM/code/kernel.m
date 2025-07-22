function K = kernel(X1, X2, type, varargin)
    switch type
        case 'linear'
            K = X1' * X2;
        case 'poly'
            p = varargin{1};
            K = (X1' * X2 + 1).^p;
        case 'rbf' % custom kernel func
            gamma = varargin{1};
            distances = pdist2(X1', X2', 'euclidean').^2;
            K = exp(-gamma * distances);
        case 'custom'
            kernel_func = varargin{1};
            K = kernel_func(X1,X2);
        otherwise
            error('Invalid kernel type: %s', type);
    end
end