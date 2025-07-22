function [X, mu, sigma] = preprocess(data, mode, mu, sigma)
    % Standardization
    if strcmp(mode, 'train')
        [X, mu, sigma]= zscore(data, 0, 2);
    else
        X = (data-mu)./sigma;
    end
end
