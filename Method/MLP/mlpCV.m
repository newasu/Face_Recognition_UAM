function [ foldLog, avgFoldLog ] = mlpCV(foldIdx, data, labels, fileNames, varargin)
%annCV Summary of this function goes here
%   Detailed explanation goes here

%     transferFunction = getAdditionalParam( 'transferFunction', varargin, 'tansig' );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    seed = getAdditionalParam( 'seed', varargin, 1 );

    noFold = size(foldIdx,1);
    
    countingRound = 0;
    finishedRound = noFold * numel(hiddenNodes);
    foldLog = [];
    for fold = 1 : noFold
        trainingData = data;
        trainingLabel = labels;
        trainingFileNames = fileNames;
        
        % train 4 out of 5 part
        trainingData(foldIdx(fold,:),:) = [];
        trainingLabel(foldIdx(fold,:),:) = [];
        trainingFileNames(foldIdx(fold,:),:) = [];
        
        % test the rest part
        testData = data(foldIdx(fold,:),:);
        testLabel = labels(foldIdx(fold,:),:);
        testFileNames = fileNames(foldIdx(fold,:),:);
        
        for i = 1 : numel(hiddenNodes)
            [~, score, mdl, label_mat, trainingTime, testTime] = mlpClassify(trainingData, trainingLabel,...
                testData, testLabel, testFileNames, 'seed', seed, 'hiddenNodes', hiddenNodes(i));
            
            % exclude model for reducing file size
            mdl = [];
            
            foldLog = [foldLog; num2cell(hiddenNodes(i)) fold score.F1_score {label_mat} {mdl} trainingTime testTime];

            % Count Progress Bar
            countingRound = countingRound + 1;
            disp([ 'Seed: '  num2str(seed) ', Finished: ' num2str(countingRound) '/' num2str(finishedRound)]);
        end
        
    end
    
    numbCVParam = 1;
    foldLog = sortrows(foldLog,[(1:numbCVParam) (numbCVParam+1)]);
    foldLog = array2table(foldLog, 'VariableNames', {'hiddenNodes', 'fold', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    % Average 5 folds into 1
    avgFoldLog = cell2mat(table2array(foldLog(:, [(1:numbCVParam) 2+numbCVParam 5+numbCVParam 6+numbCVParam ])));
    avgFoldLog = reshape(avgFoldLog',[(size(avgFoldLog,2)), noFold, (size(avgFoldLog,1)/noFold)]);
    avgFoldLog = sum(avgFoldLog,2)/noFold;
    avgFoldLog = reshape(avgFoldLog,[size(avgFoldLog,1) size(avgFoldLog,3)])';
    avgFoldLog = array2table(sortrows(avgFoldLog,-(numbCVParam+1)), 'VariableNames', {'hiddenNodes', 'score', 'trainingTime', 'testTime'});
    
    % exclude model for reducing file size
    foldLog.model = [];
end

