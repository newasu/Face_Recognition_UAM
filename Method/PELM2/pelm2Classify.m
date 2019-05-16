function [predictY, score, mdl, label_mat, trainingTime, testTime] = pelm2Classify(...
    trainingDataX_1, trainingDataX_2, trainingDataY, trainingCode, ...
    testDataX_1, testDataX_2, testDataY, testFileNames, varargin)
%PELM2CLASSIFY Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'cosine' );  % euclidean cosine
    balance = getAdditionalParam( 'balance', varargin, 1 );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    combine_rule = getAdditionalParam( 'combine_rule', varargin, 'sum' ); % sum minus multiply distance mean
    seed = getAdditionalParam( 'seed', varargin, 1 );
    select_weight_type = getAdditionalParam( 'select_weight_type', varargin, 'random_select' ); % random_select random_generate
    
    class_name = categorical(categories(trainingDataY));
    [new_trainingDataY, ~] = grp2idx(trainingDataY);
    
    [ trainingDataXX ] = combine_training_data(trainingDataX_1, trainingDataX_2, combine_rule);
    [ W, W_code ] = initHidden( hiddenNodes , trainingDataXX , seed, trainingCode, select_weight_type );
    
    tic
    [beta] = trainWELM_onevsall(trainingDataXX, new_trainingDataY, W, ...
        regularizationC, balance, distFunction);
    trainingTime = toc;
    
    [ testDataXX ] = combine_training_data(testDataX_1, testDataX_2, combine_rule);
    
    tic
    [predictY] = testWELM(testDataXX, W, beta, distFunction);
    testTime = toc;
    
%     Convert predictY back to actual label
    predictY = class_name(round(predictY));
    
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    
    label_mat = table(testFileNames, testDataY, predictY, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels'});
    
    mdl = table(W_code, beta);
end

function [ HH ] = combine_training_data(XX_1, XX_2, cr)
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
    elseif strcmp(cr, 'linear')
        HH = XX_1 * XX_2';
	elseif strcmp(cr, 'euclidean')
        HH= simKernel(XX_1, XX_2, cr);
    end
end

function [ W, W_code ] = initHidden( h , X , seed, code, weighttype )
    X_size_row = size(X,1);
    h_size = round((h/100) * X_size_row);
    h_size = min(X_size_row, h_size);
    rng(seed);
    if strcmp(weighttype, 'random_select')
        random_pick_index = randperm(size(X,1),h_size);
        W = X(random_pick_index, :);
        W_code = code(random_pick_index, :);
    else % random_generate
        W = rand(h_size, size(X,2));
        W_code = repmat(seed, h_size, size(code,2));
    end
    
end

function [beta] = trainWELM_onevsall(XX, T, W, regularizationC, balance, distFunction )

    [ H ] = simKernel(XX, W, distFunction);
    
    clear XX W
    
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
    beta = ( (H'*H) + ( (1/regularizationC) * eye(size(H,2)) ) ) \ (H'*(T_onehot.*B));
end

function [ HH ] = simKernel(XX, WW, distFunc)
	if strcmp(distFunc, 'cosine') || strcmp(distFunc, 'jaccard') || ...
            strcmp(distFunc, 'squaredeuclidean') || strcmp(distFunc, 'euclidean') % distance
        
        % divide data to calculate distance for avoiding over memory usage
%         my_gpuDevice = gpuDevice(1);
        round_step = 20000;
        ii_round = ceil(size(XX,1)/round_step);
        jj_round = ceil(size(WW,1)/round_step);
        HH = [];
        for ii = 1 : ii_round
            ii_s_idx = ((ii-1) * round_step) + 1;
            if ii == ii_round
                ii_e_idx = ((ii-1) * round_step) + mod(size(XX,1),round_step);
            else
                ii_e_idx = ii * round_step;
            end
            
            for jj = 1 : jj_round
                jj_s_idx = ((jj-1) * round_step) + 1;
                if jj == jj_round
                    jj_e_idx = ((jj-1) * round_step) + mod(size(WW,1),round_step);
                else
                    jj_e_idx = jj * round_step;
                end
                
                temp_XX = XX(ii_s_idx:ii_e_idx, :);
                temp_WW = WW(jj_s_idx:jj_e_idx, :);
                
                % gpu array if available
                if gpuDeviceCount > 0 
                    temp_XX = gpuArray(temp_XX);
                    temp_WW = gpuArray(temp_WW);
                end
                
                HH(ii_s_idx:ii_e_idx, jj_s_idx:jj_e_idx) ...
                    = gather(pdist2(temp_XX, temp_WW, distFunc));
                
                clear temp_XX temp_WW
            end
        end
        
    else % linear
        HH = gather(gpuArray(XX) * gpuArray(WW)');
    end
    
end

function oh = convert_onehot(c,nc)
    oh = zeros(1,nc);
    oh(c) = 1;
end

function [predictY] = testWELM(Xtest, WW, beta, distFunction)
    [ H ] = simKernel(Xtest, WW, distFunction);
    Hbeta = H*beta;
    [~, predictY] = max(Hbeta,[],2);
end

