function [ trainingResult, testResult, testCorrectIdx ] = TestWELMParams(...
    trainingData, trainingLabel, trainingFileNames, trainingCode,...
    testData, testLabel, testFileName, varargin)
%TESTANNPARAMS Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, numel(trainingLabel) );
    seed = getAdditionalParam( 'seed', varargin, 1 );

    % test training data
    [predictY, score, mdl, label_mat, trainingTime, testTime] = welmClassify(...
        trainingData, trainingLabel, trainingCode, trainingData, trainingLabel, trainingFileNames, 'seed', seed,...
        'distFunction', distFunction, 'hiddenNodes', hiddenNodes, 'regularizationC', regularizationC);
    trainingResult = array2table({hiddenNodes regularizationC score.F1_score label_mat mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'regC', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    % test test data
    [predictY, score, mdl, label_mat, trainingTime, testTime] = welmClassify(...
        trainingData, trainingLabel, trainingCode, testData, testLabel, testFileName, 'seed', seed,...
        'distFunction', distFunction, 'hiddenNodes', hiddenNodes, 'regularizationC', regularizationC);
    testResult = array2table({hiddenNodes regularizationC score.F1_score label_mat mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'regC', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(label_mat.labels==label_mat.predict_labels);
end

