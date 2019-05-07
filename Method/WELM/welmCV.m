function [ foldLog, avgFoldLog ] = welmCV(foldIdx, data, labels, fileNames, data_code, varargin)
%annCV Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    seed = getAdditionalParam( 'seed', varargin, 1 );

    countingRound = 0;
    paramAll = combvec(hiddenNodes,regularizationC);
    noFold = size(foldIdx,1);
    welmParam = size(paramAll,2);
    finishedRound = noFold * welmParam;
    foldLog = [];
    for fold = 1 : noFold
        trainingData = data;
        trainingLabel = labels;
%         trainingFileNames = fileNames;
        trainingCode = data_code;
        
        % train 4 out of 5 part
        trainingData(foldIdx(fold,:),:) = [];
        trainingLabel(foldIdx(fold,:),:) = [];
%         trainingFileNames(foldIdx(fold,:),:) = [];
        trainingCode(foldIdx(fold,:),:) = [];
        
        % test the rest part
        testData = data(foldIdx(fold,:),:);
        testLabel = labels(foldIdx(fold,:),:);
        testFileNames = fileNames(foldIdx(fold,:),:);
%         testCode = data_code(foldIdx(fold,:),:);
        
        for i = 1 : welmParam
            [~, accuracy, mdl, scores, trainingTime, testTime] = welmClassify(trainingData, trainingLabel,...
                trainingCode, testData, testLabel, testFileNames, 'seed', seed,...
                'distFunction', distFunction, 'hiddenNodes', paramAll(1,i), 'regularizationC', paramAll(2,i));
            
            % exclude model for reducing file size
            mdl = [];
            
            foldLog = [foldLog; paramAll(1,i) paramAll(2,i) fold accuracy {scores} {mdl} trainingTime testTime];

            % Count Progress Bar
            countingRound = countingRound + 1;
            disp([ 'Seed: '  num2str(seed) ', Finished: ' num2str(countingRound) '/' num2str(finishedRound)]);
        end
        
    end
    
    numbCVParam = size(paramAll,1);
    foldLog = sortrows(foldLog,[(1:numbCVParam) (numbCVParam+1)]);
    foldLog = array2table(foldLog, 'VariableNames', {'hiddenNodes', 'regC', 'fold', 'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    % Average 5 folds into 1
    avgFoldLog = cell2mat(table2array(foldLog(:, [(1:numbCVParam) 2+numbCVParam 5+numbCVParam 6+numbCVParam ])));
    avgFoldLog = reshape(avgFoldLog',[(size(avgFoldLog,2)), noFold, (size(avgFoldLog,1)/noFold)]);
    avgFoldLog = sum(avgFoldLog,2)/noFold;
    avgFoldLog = reshape(avgFoldLog,[size(avgFoldLog,1) size(avgFoldLog,3)])';
    avgFoldLog = array2table(sortrows(avgFoldLog,[-(numbCVParam+1) 1]), 'VariableNames', {'hiddenNodes', 'regC', 'accuracy', 'trainingTime', 'testTime'});
    
    % exclude model for reducing file size
    foldLog.model = [];
end

