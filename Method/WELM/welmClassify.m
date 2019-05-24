function [predictY, score, mdl, label_mat, trainingTime, testTime] = welmClassify(...
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
    [beta] = trainWELM_onevsall(trainingDataX, new_trainingDataY, W, regularizationC, balance, distFunction);
    trainingTime = toc;
    
    tic
    [predictY, predict_score] = testWELM(testDataX, W, beta, distFunction);
    testTime = toc;
    
%     Convert predictY back to actual label
    predictY = class_name(round(predictY));
    
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    label_mat = table(testFileNames, testDataY, predictY, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
    
    mdl = table(W_code, beta);
end

function [beta] = trainWELM_onevsall(X, T, W, regularizationC, balance, distFunction )
%     [ H ] = simKernel(X, W, distFunction);
    if gpuDeviceCount > 0 
        [ H ] = simKernel_gpu(X, W, distFunction);
    else
        [ H ] = simKernel(X, W, distFunction);
	end
    
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
	if strcmp(distFunc, 'cosine') || strcmp(distFunc, 'jaccard') || ...
            strcmp(distFunc, 'squaredeuclidean') || strcmp(distFunc, 'euclidean') % distance
        HH = pdist2(XX, WW, distFunc);
    else % linear
        HH = XX*WW';
    end
    
end

function [ HH ] = simKernel_gpu(XX, WW, distFunc)
	if strcmp(distFunc, 'cosine') || strcmp(distFunc, 'jaccard') || ...
            strcmp(distFunc, 'squaredeuclidean') || strcmp(distFunc, 'euclidean') % distance
%         HH = pdist2(gpuArray(XX), gpuArray(WW), distFunc);
        
        % divide data to calculate distance for avoiding over memory using
        my_gpuDevice = gpuDevice(1);
        ii_step = my_gpuDevice.MaxGridSize(2);
        jj_step = my_gpuDevice.MaxGridSize(3);
        ii_round = ceil(size(XX,1)/ii_step);
        jj_round = ceil(size(WW,1)/jj_step);
        HH = [];
        for ii = 1 : ii_round
            ii_s_idx = ((ii-1) * ii_step) + 1;
            if ii == ii_round
                ii_e_idx = ((ii-1) * ii_step) + mod(size(XX,1),ii_step);
            else
                ii_e_idx = ii * ii_step;
            end
            
            for jj = 1 : jj_round
                jj_s_idx = ((jj-1) * jj_step) + 1;
                if jj == jj_round
                    jj_e_idx = ((jj-1) * jj_step) + mod(size(WW,1),jj_step);
                else
                    jj_e_idx = jj * jj_step;
                end
                
                temp_XX = gpuArray(XX(ii_s_idx:ii_e_idx, :));
                temp_WW = gpuArray(WW(jj_s_idx:jj_e_idx, :));
                
                HH(ii_s_idx:ii_e_idx, jj_s_idx:jj_e_idx) ...
                    = gather(pdist2(temp_XX, temp_WW, distFunc));
                
                clear temp_XX temp_WW
            end
        end
        
    else % linear
        HH = gather(gpuArray(XX) * gpuArray(WW)');
    end
    
end

function [predictY, Hbeta] = testWELM(Xtest, WW, beta, distFunction )
    if gpuDeviceCount > 0 
        [ H ] = simKernel_gpu(Xtest, WW, distFunction);
    else
        [ H ] = simKernel(Xtest, WW, distFunction);
	end

    Hbeta = H*beta;
    [~, predictY] = max(Hbeta,[],2);
end

