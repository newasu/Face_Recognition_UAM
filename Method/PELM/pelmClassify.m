function [predictY, score, mdl, label_mat, trainingTime, testTime] = pelmClassify(...
    trainingDataX_1, trainingDataX_2, trainingDataY, trainingDataU, trainingCode,...
    testDataX_1, testDataX_2, testDataY, testFileNames, varargin)
%PELMCLASSIFY Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'cosine' );  % euclidean cosine
    balance = getAdditionalParam( 'balance', varargin, 1 );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    combine_rule = getAdditionalParam( 'combine_rule', varargin, 'sum' ); % sum minus multiply distance mean
    seed = getAdditionalParam( 'seed', varargin, 1 );
    
    class_name = categorical(categories(trainingDataY));
    [new_trainingDataY, ~] = grp2idx(trainingDataY);
    
    [ W, W_code ] = initHidden( hiddenNodes , trainingDataU , seed, trainingCode );
    
    tic
    [~, beta] = trainWELM_onevsall(trainingDataX_1, trainingDataX_2,...
        new_trainingDataY, W, regularizationC, balance, distFunction, combine_rule);
    trainingTime = toc;
    
    tic
    [predictY] = testWELM(testDataX_1, testDataX_2, W, beta, distFunction, combine_rule);
    testTime = toc;
    
%     Convert predictY back to actual label
    predictY = class_name(round(predictY));
    
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    
    label_mat = table(testFileNames, testDataY, predictY, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels'});
    
    mdl = table(W_code, beta);
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

function [W, beta] = trainWELM_onevsall(X_1, X_2, T, W, regularizationC, balance, distFunction, rule )
    [ H_1 ] = simKernel(X_1, W, distFunction);
    [ H_2 ] = simKernel(X_2, W, distFunction);
    
    [ H ] = combineRule(H_1, H_2, rule);
    
    if balance == 1
        [S,SI] = hist(T,unique(T));
        maxS = max(S);
        S = sqrt(S.\maxS);
    else
        SI = unique(T);
        S = ones(1,numel(SI));
    end
    
    T_onehot = full(ind2vec(T')');
%     T_onehot = cell2mat(arrayfun(@(x) convert_onehot(x,numel(SI)), T, 'UniformOutput', false));
    
    B = cell2mat(arrayfun(@(x) S(find(x==SI)), T, 'UniformOutput', false));
    H = repmat(B,1,size(H,2)).*H;
    beta = inv( (H'*H) + ( (1/regularizationC) * eye(size(H,2)) ) ) * (H'*(T_onehot.*B));
end

function [ HH ] = simKernel(XX, WW, distFunc)
	if gpuDeviceCount > 0 
        XX = gpuArray(XX);
        WW = gpuArray(WW);
	end
    
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
    
    if gpuDeviceCount > 0 
        HH = gather(HH);
    end
end

function [ HH ] = combineRule(XX_1, XX_2, cr)
    if strcmp(cr, 'sum')
        HH = XX_1 + XX_2;
    elseif strcmp(cr, 'minus')
        HH = XX_1 - XX_2;
    elseif strcmp(cr, 'multiply')
        HH = XX_1 .* XX_2; 
    elseif strcmp(cr, 'distance')
        HH = abs(XX_1 - XX_2);
    elseif strcmp(cr, 'mean')
        HH = (XX_1 + XX_2)/2;
    end
end

function oh = convert_onehot(c,nc)
    oh = zeros(1,nc);
    oh(c) = 1;
end

function [predictY] = testWELM(Xtest_1, Xtest_2, W, beta, distFunction, rule)
    [ H_1 ] = simKernel(Xtest_1, W, distFunction);
    [ H_2 ] = simKernel(Xtest_2, W, distFunction);
    [ H ] = combineRule(H_1, H_2, rule);
    Hbeta = H*beta;
    [~, predictY] = max(Hbeta,[],2);
end

