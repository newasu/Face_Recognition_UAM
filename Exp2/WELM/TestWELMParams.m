function [ trainingResult, testResult, testCorrectIdx ] = TestWELMParams(trainingData, trainingLabel, trainingFileNames,...
    testData, testLabel, testFileName, varargin)
%TESTANNPARAMS Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, numel(trainingLabel) );
    seed = getAdditionalParam( 'seed', varargin, 1 );

    % test training data
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = welmClassify(...
        trainingData, trainingLabel, trainingData, trainingLabel, trainingFileNames, 'seed', seed,...
        'distFunction', distFunction, 'hiddenNodes', hiddenNodes, 'regularizationC', regularizationC);
    trainingResult = array2table({hiddenNodes regularizationC accuracy scores mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'regC', 'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    % test test data
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = welmClassify(...
        trainingData, trainingLabel, testData, testLabel, testFileName, 'seed', seed,...
        'distFunction', distFunction, 'hiddenNodes', hiddenNodes, 'regularizationC', regularizationC);
    testResult = array2table({hiddenNodes regularizationC accuracy scores mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'regC', 'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(scores.labels==round(scores.predict_labels));
end

