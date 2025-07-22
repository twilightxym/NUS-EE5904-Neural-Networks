%% Load dataset and split into training set and test set
function [X,Y,X_train, Y_train, X_test, Y_test] = load_dataset(img_path)
    % Initialize image datastore with folder structure labels
    imds = imageDatastore(img_path, 'IncludeSubfolders',true,'FileExtensions', '.jpg','LabelSource', 'foldernames');
    T = imds.readall(); % Read all images and associated labels
    
    % Preprocess images: convert to grayscale and flatten to vectors
    images = cellfun(@(x) rgb2gray(x), T, 'UniformOutput', false);
    flat_images = cellfun(@(x) double(x(:)), images, 'UniformOutput', false); % Convert to column vectors
    X = cell2mat(flat_images'); % Create feature matrix (pixels × samples)
    
    % Process labels: binary encoding (1=dog, 0=automobile)
    Y = imds.Labels;
    Y = (Y == 'dog')'; % Binarize labels
    
    % Dataset partitioning parameters
    train_per_class = 450;  % Training samples per class
    test_per_class = 50;    % Test samples per class
    
    % Create index ranges for stratified sampling
    dog_train_idx = 1:train_per_class;                  % 1-450 (dog training)
    dog_test_idx = (train_per_class+1):500;             % 451-500 (dog testing)
    
    auto_train_idx = 501:500+train_per_class;           % 501-950 (automobile training)
    auto_test_idx = 500+train_per_class+1:1000;         % 951-1000 (automobile testing)
    
    % Merge indices for combined training/test sets
    trainIdx = [dog_train_idx, auto_train_idx];  % 1-450 + 501-950 → 900 samples
    testIdx = [dog_test_idx, auto_test_idx];     % 451-500 + 951-1000 → 100 samples
    
    % Create partitioned datasets
    X_train = X(:, trainIdx);  % Training features (1024×900)
    Y_train = Y(:, trainIdx);  % Training labels (1×900)
    
    X_test = X(:, testIdx);    % Test features (1024×100)
    Y_test = Y(:, testIdx);    % Test labels (1×100)
end
