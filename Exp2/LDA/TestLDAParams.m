function [ trainingResult, testResult, testCorrectIdx ] = TestLDAParams(trainingData, trainingLabel, trainingFileNames,...
    testData, testLabel, testFileName, varargin)
%TESTLDAPARAMS Summary of this function goes here
%   Detailed explanation goes here

    seed = getAdditionalParam( 'seed', varargin, 1 );
    kernelType = getAdditionalParam( 'kernelType', varargin, '' );

    % test training data
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = ldaClassify(...
        trainingData, trainingLabel, trainingData, trainingLabel, trainingFileNames, 'seed', seed,...
        'kernelType', kernelType);
    trainingResult = array2table({  accuracy scores mdl trainingTime testTime},...
        'VariableNames', {'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    % test test data
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = ldaClassify(...
        trainingData, trainingLabel, testData, testLabel, testFileName, 'seed', seed,...
        'kernelType', kernelType);
    testResult = array2table({accuracy scores mdl trainingTime testTime},...
        'VariableNames', {'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(scores.labels==round(scores.predict_labels));

end

