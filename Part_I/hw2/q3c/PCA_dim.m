function dim = PCA_dim(all_images)
% 标准化数据（逐特征）
mu = mean(all_images, 2);
sigma = std(all_images, 0, 2);
sigma(sigma==0) = 1;
all_images_z = (all_images - mu) ./ sigma;

% PCA分析确定隐藏层神经元数量
[~, ~, ~, ~, explained] = pca(all_images_z');
cumulative_variance = cumsum(explained);
retain_components = find(cumulative_variance >= 95, 1);
fprintf('PCA建议隐藏层神经元数量: %d (保留%.1f%%方差)\n', retain_components, ...
    cumulative_variance(retain_components));
dim = retain_components;
end