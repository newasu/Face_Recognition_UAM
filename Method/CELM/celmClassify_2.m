function [predictY, score, mdl, label_mat, trainingTime, testTime] = celmClassify_2(...
    trainingDataX_1, trainingDataX_2, trainingDataY, trainingCode, ...
    testDataX_1, testDataX_2, testDataY, testFileNames, data_feature, varargin)
%CELMCLASSIFY Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    balance = getAdditionalParam( 'balance', varargin, 1 );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    seed = getAdditionalParam( 'seed', varargin, 1 );
    select_weight_type = getAdditionalParam( 'select_weight_type', varargin, 'randomselect' ); % randomselect randomgenerate
    
    class_name = categorical(categories(trainingDataY));
    [new_trainingDataY, ~] = grp2idx(trainingDataY);
    
    [ trainingDataXX ] = combine_training_data(data_feature(trainingDataX_1,:), data_feature(trainingDataX_2,:));
    [ W, W_code ] = initHidden( hiddenNodes , trainingDataXX , seed, trainingCode, select_weight_type );
    
    tic
    [beta] = trainWELM_onevsall(trainingDataXX, new_trainingDataY, W, ...
        regularizationC, balance, distFunction);
    trainingTime = toc;
    clear trainingDataXX trainingDataX_1 trainingDataX_2
    
    tic
    [predictY, predict_score] = testWELM(testDataX_1, testDataX_2, data_feature, W, beta, distFunction);
    testTime = toc;
    
%     Convert predictY back to actual label
    predictY = class_name(round(predictY));
    
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    
    % AUC
    auc = [];
    for ii = 1 : numel(class_name)
        temp = testDataY==class_name(ii);
        [~,~,~,auc(ii)] = perfcurve(temp, predict_score(:,ii),1);
    end
    score.AUC = sum(auc)/numel(class_name);
    
    % biometric score
    [biometric_perf_threshold, biometric_perf_mat] = exp3_report_biometric_perf(...
        testDataY, predictY, predict_score(:, (class_name == 'same')), 'same');
    score.EER = biometric_perf_mat.EER;
    score.FMR_0d1 = biometric_perf_mat.FMR_0d1;
    score.FMR_0d01 = biometric_perf_mat.FMR_0d01;
    
    label_mat = table(testFileNames, testDataY, predictY, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
    
    mdl = table(W_code, beta);

end

function [ HH ] = combine_training_data(XX_1, XX_2)
    HH = [XX_1, XX_2];
end

function [ W, W_code ] = initHidden( h , X , seed, code, weighttype )
    X_size_row = size(X,1);
    h_size = round((h/100) * X_size_row);
    h_size = min(X_size_row, h_size);
    SetRandomSeed(seed);
    if strcmp(weighttype, 'randomselect')
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
    gpu_round_step = 18000;
    
    if strcmp(distFunc, 'euclidean') && gpuDeviceCount > 0 ...
            && size(XX,1) <= gpu_round_step && size(WW,1) <= gpu_round_step
        XXXX = (sum(gpuArray(XX).^2, 2));
        XXWW = (gpuArray(XX) * gpuArray(WW)');
        WWWW = (sum(gpuArray(WW).^2, 2)');
        HH = bsxfun(@minus, XXXX, 2 * XXWW);
        clear XXXX XXWW
        HH = bsxfun(@plus, WWWW, HH);
        clear WWWW
        HH = gather(HH);
        HH = sqrt(HH);
        HH = real(HH);
        
    elseif strcmp(distFunc, 'cosine') || strcmp(distFunc, 'jaccard') || ...
        strcmp(distFunc, 'squaredeuclidean') || strcmp(distFunc, 'euclidean') % distance
        
        if gpuDeviceCount > 0
            gpuDevice(1);
            round_step = gpu_round_step;
        else
            round_step = 100000000000;
        end
        
        % divide data to calculate distance for avoiding over memory usage
        ii_round = ceil(size(XX,1)/round_step);
        jj_round = ceil(size(WW,1)/round_step);
        XX_size = size(XX,1);
        HH = [];
        for ii = 1 : ii_round
            
            if ii_round > 1
                disp(['Calculating kernel: ' num2str(ii) '/' num2str(ii_round)]);
            end
            
            ii_s_idx = ((ii-1) * round_step) + 1;
            if ii == ii_round
                if mod(XX_size,round_step) == 0
                    ii_e_idx = ((ii) * round_step);
                else
                    ii_e_idx = ((ii-1) * round_step) + mod(XX_size,round_step);
                end
                
                temp_XX = XX;
                XX = [];

            else
                ii_e_idx = ii * round_step;
                temp_XX = XX(1:round_step,:);
                XX(1:round_step,:) = [];

            end
            
            for jj = 1 : jj_round
                jj_s_idx = ((jj-1) * round_step) + 1;
                if jj == jj_round
                    if mod(size(WW,1),round_step) == 0
                        jj_e_idx = ((jj) * round_step);
                    else
                        jj_e_idx = ((jj-1) * round_step) + mod(size(WW,1),round_step);
                    end

                else
                    jj_e_idx = jj * round_step;
                end
                
                temp_WW = WW(jj_s_idx:jj_e_idx, :);
                
                % gpu array if available
                if gpuDeviceCount > 0 
                    temp_XX = gpuArray(temp_XX);
                    temp_WW = gpuArray(temp_WW);
                    
                    HH(ii_s_idx:ii_e_idx, jj_s_idx:jj_e_idx) ...
                        = gather(pdist2(temp_XX, temp_WW, distFunc));
                else
                    HH(ii_s_idx:ii_e_idx, jj_s_idx:jj_e_idx) ...
                        = pdist2(temp_XX, temp_WW, distFunc);
                end
                
                clear temp_WW
            end
            
            clear temp_XX
        end

    else % linear
        if gpuDeviceCount > 0
            XX = gpuArray(XX);
            WW = gpuArray(WW);
        end
        HH = gather(XX * WW');
    end
    
end

function [predictY, Hbeta] = testWELM(Xtest_1, Xtest_2, X_feature, WW, beta, distFunction)   
    Xtest_split_size = 54000;
    
    Xtest_size = numel(Xtest_1);
    Hbeta = [];
    for iii = 1 : ceil(Xtest_size/Xtest_split_size)
        temp_s_idx = ((iii-1) * Xtest_split_size) + 1;
        
        if iii == ceil(Xtest_size/Xtest_split_size)
            if mod(Xtest_size,Xtest_split_size) == 0
                temp_e_idx = ((iii) * Xtest_split_size);
            else
                temp_e_idx = ((iii-1) * Xtest_split_size) + mod(Xtest_size,Xtest_split_size);
            end
            
        else
            temp_e_idx = ((iii) * Xtest_split_size);
        end

        Xtest = combine_training_data(X_feature(Xtest_1(temp_s_idx:temp_e_idx),:), ...
            X_feature(Xtest_2(temp_s_idx:temp_e_idx),:));
    
        H = simKernel(Xtest, WW, distFunction);
        Hbeta = [Hbeta; (H*beta)];
    end

    [~, predictY] = max(Hbeta,[],2);
end

