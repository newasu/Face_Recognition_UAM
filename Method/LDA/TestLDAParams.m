function [ trainingResult, testResult, testCorrectIdx ] = TestLDAParams(...
    trainingData, trainingLabel, trainingFileNames,...
    testData, testLabel, testFileName, varargin)
%TESTLDAPARAMS Summary of this function goes here
%   Detailed explanation goes here

%     seed = getAdditionalParam( 'seed', varargin, 1 );
    kernelType = getAdditionalParam( 'kernelType', varargin, 'linear' );

    % test training data
    [predictY, score, mdl, label_mat, trainingTime, testTime] = ldaClassify(...
        trainingData, trainingLabel, trainingData, trainingLabel, trainingFileNames,...
        'kernelType', kernelType);
    trainingResult = array2table({score.F1_score label_mat mdl trainingTime testTime},...
        'VariableNames', {'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    % test test data
    [predictY, score, mdl, label_mat, trainingTime, testTime] = ldaClassify(...
        trainingData, trainingLabel, testData, testLabel, testFileName,...
        'kernelType', kernelType);
    testResult = array2table({score.F1_score label_mat mdl trainingTime testTime},...
        'VariableNames', {'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(label_mat.labels==label_mat.predict_labels);

end

