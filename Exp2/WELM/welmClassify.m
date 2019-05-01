function [predictY, accuracy, mdl, scores, trainingTime, testTime] = welmClassify(...
    trainingDataX, trainingDataY, testDataX, testDataY, testFileNames, varargin)
%ANNCLASSIFY Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'cosine' );  % euclidean cosine
    balance = getAdditionalParam( 'balance', varargin, 1 );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    seed = getAdditionalParam( 'seed', varargin, 1 );
    
    uniqueClass = unique(trainingDataY);
    numbClass = numel(uniqueClass);
    class_num = 1:numbClass;
    new_trainingDataY = cell2mat(arrayfun(@(x) class_num(find(x==uniqueClass)), trainingDataY, 'UniformOutput', false));
    
    [ W ] = initHidden( hiddenNodes , trainingDataX , seed );
    beta = [];
    
    tic
    [~, beta] = trainWELM_onevsall(trainingDataX, new_trainingDataY, W, regularizationC, balance, distFunction);
    trainingTime = toc;
    
    tic
    [predictY] = testWELM(testDataX, W, beta, distFunction);
    testTime = toc;
    
%     Convert predictY back to actual label
    predictY = cell2mat(arrayfun(@(x) uniqueClass(find(x==class_num)), predictY, 'UniformOutput', false));
    
    accuracy = (sum(round(predictY)==testDataY)/numel(testDataY)) * 100;
    scores = table(testFileNames, testDataY, predictY, 'VariableNames', {'filenames' 'labels', 'predict_labels'});
    
    mdl = table(W, beta);
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
    
    T_onehot = cell2mat(arrayfun(@(x) convert_onehot(x,numel(SI)), T, 'UniformOutput', false));
    
    B = cell2mat(arrayfun(@(x) S(find(x==SI)), T, 'UniformOutput', false));
    H = repmat(B,1,size(H,2)).*H;
    beta = inv( (H'*H) + ( (1/regularizationC) * eye(size(H,2)) ) ) * (H'*(T_onehot.*B));
end

function oh = convert_onehot(c,nc)
    oh = zeros(1,nc);
    oh(c) = 1;
end

function [ W ] = initHidden( h , X , seed )
    X_size = size(X,1);
    h_size = round((h/100) * X_size);
    h_size = min(X_size, h_size);
    rng(seed);
    W = X(randperm(size(X,1),h_size),:);
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

