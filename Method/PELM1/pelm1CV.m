function [ foldLog, avgFoldLog ] = pelm1CV(foldIdx, data_0, data_1, data_2, labels, data_code, varargin)
%PELM1CV Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    combine_rule = getAdditionalParam( 'combine_rule', varargin, 'sum' ); % sum minus multiply distance mean
    seed = getAdditionalParam( 'seed', varargin, 1 );
    select_weight_type = getAdditionalParam( 'select_weight_type', varargin, 'random_select' ); % random_select random_generate
    
    countingRound = 0;
    paramAll = combvec(hiddenNodes,regularizationC);
    noFold = size(foldIdx,1);
    welmParam = size(paramAll,2);
    finishedRound = noFold * welmParam;
    foldLog = [];
    for fold = 1 : noFold
        trainingData_1 = data_1;
        trainingData_2 = data_2;
        trainingLabel = labels;
%         trainingFileNames = fileNames;
        trainingCode = data_code;
        
        % train 4 out of 5 part
        trainingData_1(foldIdx(fold,:),:) = [];
        trainingData_2(foldIdx(fold,:),:) = [];
        trainingLabel(foldIdx(fold,:),:) = [];
%         trainingFileNames(foldIdx(fold,:),:) = [];
        trainingCode(foldIdx(fold,:),:) = [];
        trainingCode = unique(trainingCode);
        training_u = data_0(trainingCode,:);
        
        % test the rest part
        testData_1 = data_1(foldIdx(fold,:),:);
        testData_2 = data_2(foldIdx(fold,:),:);
        testLabel = labels(foldIdx(fold,:),:);
        testFileName = data_code(foldIdx(fold,:),:);
%         testCode = data_code(foldIdx(fold,:),:);
        
        for i = 1 : welmParam
            [~, score, mdl, label_mat, trainingTime, testTime] = pelm1Classify(...
                trainingData_1, trainingData_2, trainingLabel, training_u,...
                trainingCode, testData_1, testData_2, testLabel, testFileName,...
                'seed', seed, 'combine_rule', combine_rule,...
                'distFunction', distFunction, 'hiddenNodes', paramAll(1,i),...
                'regularizationC', paramAll(2,i), 'select_weight_type', select_weight_type);
            
            % exclude model for reducing file size
%             mdl = [];
            
            foldLog = [foldLog; paramAll(1,i) paramAll(2,i) fold score.Accuracy {label_mat} {mdl} trainingTime testTime];

            % Count Progress Bar
            countingRound = countingRound + 1;
            disp([ 'Seed: '  num2str(seed) ', Finished: ' num2str(countingRound) '/' num2str(finishedRound)]);
        end
        
    end
    
    numbCVParam = size(paramAll,1);
    foldLog = sortrows(foldLog,[(1:numbCVParam) (numbCVParam+1)]);
    foldLog = array2table(foldLog, 'VariableNames', {'hiddenNodes', 'regC', 'fold', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    % Average 5 folds into 1
    avgFoldLog = cell2mat(table2array(foldLog(:, [(1:numbCVParam) 2+numbCVParam 5+numbCVParam 6+numbCVParam ])));
    avgFoldLog = reshape(avgFoldLog',[(size(avgFoldLog,2)), noFold, (size(avgFoldLog,1)/noFold)]);
    avgFoldLog = sum(avgFoldLog,2)/noFold;
    avgFoldLog = reshape(avgFoldLog,[size(avgFoldLog,1) size(avgFoldLog,3)])';
    avgFoldLog = array2table(sortrows(avgFoldLog,[-(numbCVParam+1) 1]), 'VariableNames', {'hiddenNodes', 'regC', 'score', 'trainingTime', 'testTime'});
    
    % exclude model for reducing file size
    foldLog.model = [];
end

