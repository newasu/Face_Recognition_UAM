function [predictY, accuracy, mdl, scores, trainingTime, testTime] = welmClassify(...
    trainingDataX, trainingDataY, trainingCode, testDataX, testDataY, testFileNames, varargin)
%WELMClassify Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'cosine' );  % euclidean cosine
    balance = getAdditionalParam( 'balance', varargin, 1 );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    seed = getAdditionalParam( 'seed', varargin, 1 );
    
    class_name = categorical(categories(trainingDataY));
    [new_trainingDataY, ~] = grp2idx(trainingDataY);

    [ W, W_code ] = initHidden( hiddenNodes , trainingDataX , seed, trainingCode );

    tic
    [~, beta] = trainWELM_onevsall(trainingDataX, new_trainingDataY, W, regularizationC, balance, distFunction);
    trainingTime = toc;
    
    tic
    [predictY] = testWELM(testDataX, W, beta, distFunction);
    testTime = toc;
    
%     Convert predictY back to actual label
    predictY = class_name(round(predictY));
    
    accuracy = (sum(predictY==testDataY)/numel(testDataY)) * 100;
    scores = table(testFileNames, testDataY, predictY, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels'});
    
    mdl = table(W_code, beta);
end

function [W, beta] = trainWELM_onevsall(X, T, W, regularizationC, balance, distFunction )
    [ H ] = simKernel(X, W, distFunction);
    
    if balance == 1
        [S,SI] = hist(T,unique(T));
        maxS = max(S);
        S = sqrt(S.\maxS);
    else
        SI = unique(T);
        S = ones(1,numel(SI));
    end
    
%     T_onehot = cell2mat(arrayfun(@(x) convert_onehot(x,numel(SI)), T, 'UniformOutput', false));
    T_onehot = full(ind2vec(double(T)')');
    
    B = cell2mat(arrayfun(@(x) S(find(x==SI)), T, 'UniformOutput', false));
    H = repmat(B,1,size(H,2)).*H;
    beta = inv( (H'*H) + ( (1/regularizationC) * eye(size(H,2)) ) ) * (H'*(T_onehot.*B));
end

function oh = convert_onehot(c,nc)
    oh = zeros(1,nc);
    oh(c) = 1;
end

function [ W, W_code ] = initHidden( h , X , seed, code )
    X_size = size(X,1);
    h_size = round((h/100) * X_size);
    h_size = min(X_size, h_size);
    rng(seed);
    random_pick_index = randperm(size(X,1),h_size);
    W = X(random_pick_index, :);
    W_code = code(random_pick_index, :);
end

function [ HH ] = simKernel(XX, WW, distFunc)
    if strcmp(distFunc, 'cosine')
        HH = pdist2(XX,WW,'cosine');
    elseif strcmp(distFunc, 'jaccard')
        HH = pdist2(XX,WW,'jaccard');
    elseif strcmp(distFunc, 'squaredeuclidean')
        HH = pdist2(XX,WW,'squaredeuclidean');
    elseif strcmp(distFunc, 'linear')
        HH = XX*WW';
    else
        HH = pdist2(XX,WW,'euclidean');
    end
    HH = double(HH);
end

function [predictY] = testWELM(Xtest, W, beta, distFunction )
    [ H ] = simKernel(Xtest, W, distFunction);
    Hbeta = H*beta;
    [~, predictY] = max(Hbeta,[],2);
end

