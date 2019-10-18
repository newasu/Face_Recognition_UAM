function [predictY, score, mdl, label_mat, trainingTime, testTime] = pelm2Classify(...
    trainingDataX_1, trainingDataX_2, trainingDataY, trainingCode, ...
    testDataX_1, testDataX_2, testDataY, testFileNames, varargin)
%PELM2CLASSIFY Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine euclideanmm
    balance = getAdditionalParam( 'balance', varargin, 1 );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    combine_rule = getAdditionalParam( 'combine_rule', varargin, 'sum' ); % sum minus multiply distance mean
    seed = getAdditionalParam( 'seed', varargin, 1 );
    select_weight_type = getAdditionalParam( 'select_weight_type', varargin, 'randomselect' ); % randomselect randomgenerate
    
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
    [predictY, predict_score] = testWELM(testDataXX, W, beta, distFunction);
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
        testDataY, predictY, predict_score(:, find(class_name == 'same')), 'same');
    score.EER = biometric_perf_mat.EER;
    score.FMR_0d1 = biometric_perf_mat.FMR_0d1;
    score.FMR_0d01 = biometric_perf_mat.FMR_0d01;
    
    label_mat = table(testFileNames, testDataY, predictY, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
    
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
    end
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
    else % random-generate
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
    gpu_round_step = 24000;
    
    if strcmp(distFunc, 'euclidean') || gpuDeviceCount > 0 || size(XX,1) <= gpu_round_step
        XXXX = sum(XX.^2, 2);
        XXWW = XX * WW';
        WWWW = sum(WW.^2, 2)';
        HH = sqrt(bsxfun(@plus, WWWW, bsxfun(@minus, XXXX, 2 * XXWW)));
        HH = real(HH);
        
    elseif strcmp(distFunc, 'cosine') || strcmp(distFunc, 'jaccard') || ...
        strcmp(distFunc, 'squaredeuclidean') || strcmp(distFunc, 'euclidean') % distance
        
        if gpuDeviceCount > 0
%             my_gpuDevice = gpuDevice(1);
%             round_step = round(sqrt(my_gpuDevice.AvailableMemory*0.25));
            round_step = gpu_round_step;
        else
            round_step = 100000000000;
        end
        
        % divide data to calculate distance for avoiding over memory usage
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
                    
                    HH(ii_s_idx:ii_e_idx, jj_s_idx:jj_e_idx) ...
                        = gather(pdist2(temp_XX, temp_WW, distFunc));
                else
                    HH(ii_s_idx:ii_e_idx, jj_s_idx:jj_e_idx) ...
                        = pdist2(temp_XX, temp_WW, distFunc);
                end
                
                
                
                clear temp_XX temp_WW
            end
            
            if ii_round > 1
                disp(['Calculating kernel: ' num2str(ii) '/' num2str(ii_round)]);
            end
            
        end
        
%     elseif strcmp(distFunc, 'euclideanmm')
%         disp('Calculating Euclidean mm kernel..');
%         
%         % gpu array if available
%         if gpuDeviceCount == 0
%         	XXXX = sum(XX.^2, 2);
%             XXWW = XX * WW';
%             WWWW = sum(WW.^2, 2)';
%             HH = sqrt(bsxfun(@plus, WWWW, bsxfun(@minus, XXXX, 2 * XXWW)));
%             HH = real(HH);
%          
%         else
%             round_step = 20000;
%             
%             XXXX = [];
%             for iii = 1 : ceil(size(XX,1)/round_step)
%                 % start idx
%                 start_idx = ((iii-1) * round_step) + 1;
%                 if (start_idx + round_step - 1) < size(XX,1)
%                     start_idx = start_idx : (start_idx + round_step - 1);
%                 else
%                     start_idx = start_idx : (start_idx + mod(size(XX,1),round_step) - 1);
%                 end
% 
%                 % calculate
%                 temp = gpuArray(XX(start_idx,:));
% %                 % gpu array if available
% %                 if gpuDeviceCount > 0
% %                     temp = gpuArray(temp);
% %                 end
%                 XXXX(start_idx, 1) = gather(sum(temp.^2, 2));
%                 clear temp
%             end
% 
%             XXWW = [];
%             for iii = 1 : size(XX,1)
%                 for jjj = 1 : ceil(size(WW,1)/round_step)
%                     % start idx
%                     start_idx = ((jjj-1) * round_step) + 1;
%                     if (start_idx + round_step - 1) < size(WW,1)
%                         start_idx = start_idx : (start_idx + round_step - 1);
%                     else
%                         start_idx = start_idx : (start_idx + mod(size(WW,1),round_step) - 1);
%                     end
% 
%                     % calculate
%                     temp_XX = gpuArray(XX(iii,:));
%                     temp_WW = gpuArray(WW(start_idx,:));
% %                     % gpu array if available
% %                     if gpuDeviceCount > 0
% %                         temp_XX = gpuArray(temp_XX);
% %                         temp_WW = gpuArray(temp_WW);
% %                     end
%                     XXWW(iii,start_idx) = gather(temp_XX * temp_WW');
%                     clear temp_XX temp_WW
%                 end
%             end
% 
%             WWWW = [];
%             for iii = 1 : ceil(size(WW,1)/round_step)
%                 % start idx
%                 start_idx = ((iii-1) * round_step) + 1;
%                 if (start_idx + round_step - 1) < size(WW,1)
%                     start_idx = start_idx : (start_idx + round_step - 1);
%                 else
%                     start_idx = start_idx : (start_idx + mod(size(WW,1),round_step) - 1);
%                 end
% 
%                 % calculate
%                 temp = gpuArray(WW(start_idx,:));
% %                 % gpu array if available
% %                 if gpuDeviceCount > 0
% %                     temp = gpuArray(temp);
% %                 end
%                 WWWW(start_idx) = gather(sum(temp.^2, 2)');
%                 clear temp
%             end
% 
%             XXXX = gpuArray(XXXX);
%             WWWW = gpuArray(WWWW);
% %             if gpuDeviceCount > 0
% %                 XXXX = gpuArray(XXXX);
% %                 WWWW = gpuArray(WWWW);
% %             end
% 
%             HH = [];
%             for iii = 1 : size(XXWW,1)
%                 temp = gpuArray(XXWW(iii,:));
% %                 if gpuDeviceCount > 0
% %                     temp = gpuArray(temp);
% %                 end
% 
%                 temp = sqrt( complex( WWWW + (XXXX(iii) - (2*temp)) ) );
%                 
%                 temp = gather(temp);
% %                 if gpuDeviceCount > 0
% %                     temp = gather(temp);
% %                 end
% 
%                 HH(iii,:) = temp;
%             end
%             HH = real(HH);
%             clear XXXX WWWW temp
%         end
%         
    else % linear
        if gpuDeviceCount > 0
            XX = gpuArray(XX);
            WW = gpuArray(WW);
        end
        HH = gather(XX * WW');
    end
    
end

function oh = convert_onehot(c,nc)
    oh = zeros(1,nc);
    oh(c) = 1;
end

function [predictY, Hbeta] = testWELM(Xtest, WW, beta, distFunction)
    [ H ] = simKernel(Xtest, WW, distFunction);
    Hbeta = H*beta;
    [~, predictY] = max(Hbeta,[],2);
end

