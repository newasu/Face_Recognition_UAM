function [ trainingResult, testResult, testCorrectIdx ] = TestSVMParams(...
    trainingData, trainingLabel, trainingFileNames,...
    testData, testLabel, testFileName, optParam, paramName, varargin)
%TESTSVMPARAMS Summary of this function goes here
%   Detailed explanation goes here
    
    kernelType = getAdditionalParam( 'kernelType', varargin, 'linear' );
    svmType = getAdditionalParam( 'svmType', varargin, 0 ); % 0 for C-SVC, 2 for one-class SVM
    libsvmKernelType = getAdditionalParam( 'libsvmKernelType', varargin, 1 ); % 0 for linear, 2 for rbf, 4 for precomputed
    
    % Do kernel
    if libsvmKernelType == 4
        if contains(kernelType,'rbf') % do rbf
            trainingKernel = kernel(trainingData, trainingData, kernelType, 'gamma', optParam.g);
            testKernel = kernel(testData, trainingData, kernelType, 'gamma',  optParam.g);
        else % do linear
            trainingKernel = kernel(trainingData, trainingData, kernelType);
            testKernel = kernel(testData, trainingData, kernelType);
        end
    else % dont use precomputed
        trainingKernel = trainingData;
        testKernel = testData;
    end
    
    % test training datas
    svmCommand = buildSVMString('training', 'training', {'s' 't' paramName{:}}, [svmType libsvmKernelType table2array(optParam)]);
    eval(svmCommand);
    trainingResult = array2table({table2array(optParam) score.F1_score label_mat mdl trainingTime testTime},...
        'VariableNames', {paramName{:}, 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    % test test data
    svmCommand = buildSVMString('training', 'test', {'s' 't' paramName{:}}, [svmType libsvmKernelType table2array(optParam)]);
    eval(svmCommand);
    testResult = array2table({table2array(optParam) score.F1_score label_mat mdl trainingTime testTime},...
        'VariableNames', {paramName{:}, 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(label_mat.labels==label_mat.predict_labels);
end

function svmCommand = buildSVMString(trainingStr, testStr, svmParamName, svmParam)
    svmCommand = ['[~, score, mdl, label_mat, trainingTime, testTime] = libsvmClassify('...
        selectDataUsed(trainingStr) selectDataUsed(testStr) selectFileNameUsed(testStr)];
    
    for i = 1 : numel(svmParamName)
        svmCommand = [ svmCommand ', ''' svmParamName{i} ''', ' num2str(svmParam(i))];
    end
    svmCommand = [svmCommand ');'];
end

function dataUsedStr = selectDataUsed(dt)
    if strcmp(dt,'training')
        dataUsedStr = 'trainingKernel, trainingLabel, ';
    else
        dataUsedStr = 'testKernel, testLabel, ';
    end
end

function fnUsedStr = selectFileNameUsed(fn)
    if strcmp(fn,'training')
        fnUsedStr = 'trainingFileNames';
    else
        fnUsedStr = 'testFileName';
    end
end