function [foldLog, avgFoldLog] = celmCV_2(foldIdx, data_1, data_2, data_id, data_label, data_feature, varargin)
%CELMCV Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    seed = getAdditionalParam( 'seed', varargin, 1 );
    select_weight_type = getAdditionalParam( 'select_weight_type', varargin, 'randomselect' ); % randomselect randomgenerate
    
    countingRound = 0;
    paramAll = combvec(hiddenNodes,regularizationC);
    noFold = size(foldIdx,1);
    welmParam = size(paramAll,2);
    finishedRound = noFold * welmParam;
    foldLog = [];
    for fold = 1 : noFold
        trainingData_1 = data_1;
        trainingData_2 = data_2;
        trainingLabel = data_label;
%         trainingFileNames = fileNames;
        training_data_id = data_id;
        
        % train 4 out of 5 part
        trainingData_1(foldIdx(fold,:),:) = [];
        trainingData_2(foldIdx(fold,:),:) = [];
        trainingLabel(foldIdx(fold,:),:) = [];
%         trainingFileNames(foldIdx(fold,:),:) = [];
        training_data_id(foldIdx(fold,:),:) = [];
        
        % test the rest part
        testData_1 = data_1(foldIdx(fold,:),:);
        testData_2 = data_2(foldIdx(fold,:),:);
        testLabel = data_label(foldIdx(fold,:),:);
        testFileNames = data_id(foldIdx(fold,:),:);
%         testCode = data_code(foldIdx(fold,:),:);
        
        for i = 1 : welmParam
            [~, score, mdl, label_mat, trainingTime, testTime] = celmClassify_2(...
                trainingData_1, trainingData_2, trainingLabel, training_data_id, ...
                testData_1, testData_2, testLabel, testFileNames, data_feature, ...
                'seed', seed, 'regularizationC', paramAll(2,i), ...
                'distFunction', distFunction, 'hiddenNodes', paramAll(1,i), ...
                'select_weight_type', select_weight_type);
            
            % exclude model for reducing file size
%             mdl = [];
            
            foldLog = [foldLog; paramAll(1,i) paramAll(2,i) fold ...
                score.FMR_0d01 score.AUC score.EER score.FMR_0d1 ...
                {label_mat} {mdl} trainingTime testTime];

            % Count Progress Bar
            countingRound = countingRound + 1;
            disp([ 'Seed: '  num2str(seed) ', Finished: ' num2str(countingRound) '/' num2str(finishedRound)]);
        end
        
    end
    
    numbCVParam = size(paramAll,1);
    foldLog = sortrows(foldLog,[(1:numbCVParam) (numbCVParam+1)]);
    foldLog = array2table(foldLog, 'VariableNames', ...
        {'hiddenNodes', 'regC', 'fold', ...
        'score_FMR_0d01', 'score_AUC', 'score_EER', 'score_FMR_0d1', ...
        'label_mat', 'model', 'trainingTime', 'testTime'});
    
    % Average 5 folds into 1
    avgFoldLog = cell2mat(table2array(foldLog(:, (1:numbCVParam))));
    avgFoldLog = [avgFoldLog cell2mat(table2array(foldLog(:, {'score_FMR_0d01', 'score_AUC', 'score_EER', 'score_FMR_0d1', 'trainingTime', 'testTime'})))];
    avgFoldLog = reshape(avgFoldLog',[(size(avgFoldLog,2)), noFold, (size(avgFoldLog,1)/noFold)]);
    avgFoldLog = sum(avgFoldLog,2)/noFold;
    avgFoldLog = reshape(avgFoldLog,[size(avgFoldLog,1) size(avgFoldLog,3)])';
    avgFoldLog = array2table(sortrows(avgFoldLog,[(numbCVParam+1) 1:numbCVParam]), ...
        'VariableNames', {'hiddenNodes', 'regC', ...
        'score_FMR_0d01', 'score_AUC', 'score_EER', 'score_FMR_0d1', ...
        'trainingTime', 'testTime'});
    
    % exclude model for reducing file size
    foldLog.model = [];

end

