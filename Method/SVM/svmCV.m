function [ foldLog, avgFoldLog ] = svmCV(foldIdx, cvParam, cvParamName,...
    data, labels, fileNames, varargin)
%SVMCV Summary of this function goes here
%   Detailed explanation goes here

    kernelType = getAdditionalParam( 'kernelType', varargin, 'linear' );
    svmType = getAdditionalParam( 'svmType', varargin, 0 ); % 0 for C-SVC, 2 for one-class SVM
    libsvmKernelType = getAdditionalParam( 'libsvmKernelType', varargin, 1 ); % 0 for linear, 2 for rbf, 4 for precomputed
    seed = getAdditionalParam( 'seed', varargin, 1 );
    
    paramAll = combvec(cvParam{:});
    noFold = size(foldIdx,1);
    gIdx = find(strcmp(cvParamName,'g')); % find gamma index
    kernelFlag = nan;
    
    countingRound = 0;
    finishedRound = noFold * size(paramAll,2);
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
        
        for i = 1 : size(paramAll,2)
            % Check for avoiding to unnecessary do kernel
            if libsvmKernelType == 4 % Precomputed
                if contains(kernelType,'rbf') % do rbf
                    if kernelFlag ~= paramAll(gIdx,i)
                        trainingKernel = kernel(trainingData, trainingData, kernelType, 'gamma',  paramAll(gIdx,i));
                        testKernel = kernel(testData, trainingData, kernelType, 'gamma',  paramAll(gIdx,i));
                        kernelFlag = paramAll(gIdx,i);
                    end
                else % do linear
                    if isnan(kernelFlag)
                        trainingKernel = kernel(trainingData, trainingData, kernelType);
                        testKernel = kernel(testData, trainingData, kernelType);
                        kernelFlag = 1; % done kernel
                    end
                end
            else % do kernel in libsvm (don't use precomputed)
                if isnan(kernelFlag)
                    trainingKernel = trainingData;
                    testKernel = testData;
                    kernelFlag = 1;
                end
            end
            
            if numel(cvParamName) == 1
                [~, score, mdl, label_mat, trainingTime, testTime] = libsvmClassify(trainingKernel, trainingLabel,...
                    testKernel, testLabel, testFileNames, 's', svmType, 't', libsvmKernelType, cvParamName{1}, paramAll(1,i));
            else
                [~, score, mdl, label_mat, trainingTime, testTime] = libsvmClassify(trainingKernel, trainingLabel,...
                    testKernel, testLabel, testFileNames, 's', svmType, 't', libsvmKernelType, cvParamName{1}, paramAll(1,i), cvParamName{2}, paramAll(2,i));
            end
            
            foldLog = [foldLog; num2cell(paramAll(:,i)') fold score.F1_score {label_mat} {mdl} trainingTime testTime];

            % Count Progress Bar
            countingRound = countingRound + 1;
            disp([ 'Seed: '  num2str(seed) ', Finished: ' num2str(countingRound) '/' num2str(finishedRound)]);
        end
        
        kernelFlag = nan;
    end
    
    numbCVParam = numel(cvParamName);
    foldLog = sortrows(foldLog,[(1:numbCVParam) (numbCVParam+1)]);
    foldLog = array2table(foldLog, 'VariableNames', {cvParamName{:}, 'fold', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    % Average 5 folds into 1
    avgFoldLog = cell2mat(table2array(foldLog(:, [(1:numel(cvParamName)) 2+numbCVParam 5+numbCVParam 6+numbCVParam ])));
    avgFoldLog = reshape(avgFoldLog',[(size(avgFoldLog,2)), noFold, (size(avgFoldLog,1)/noFold)]);
    avgFoldLog = sum(avgFoldLog,2)/noFold;
    avgFoldLog = reshape(avgFoldLog,[size(avgFoldLog,1) size(avgFoldLog,3)])';
    avgFoldLog = array2table(sortrows(avgFoldLog,-(numbCVParam+1)), 'VariableNames', {cvParamName{:}, 'score', 'trainingTime', 'testTime'});
end

