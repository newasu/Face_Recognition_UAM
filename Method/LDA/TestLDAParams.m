function [ trainingResult, testResult, testCorrectIdx ] = TestLDAParams(...
    trainingData, trainingLabel, trainingFileNames,...
    testData, testLabel, testFileName, varargin)
%TESTLDAPARAMS Summary of this function goes here
%   Detailed explanation goes here

%     seed = getAdditionalParam( 'seed', varargin, 1 );
    kernelType = getAdditionalParam( 'kernelType', varargin, 'linear' );

    % test training data
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = ldaClassify(...
        trainingData, trainingLabel, trainingData, trainingLabel, trainingFileNames,...
        'kernelType', kernelType);
    trainingResult = array2table({  accuracy scores mdl trainingTime testTime},...
        'VariableNames', {'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    % test test data
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = ldaClassify(...
        trainingData, trainingLabel, testData, testLabel, testFileName,...
        'kernelType', kernelType);
    testResult = array2table({accuracy scores mdl trainingTime testTime},...
        'VariableNames', {'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(scores.labels==scores.predict_labels);

end

