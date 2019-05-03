function [ trainingResult, testResult, testCorrectIdx ] = TestANNParams(...
    trainingData, trainingLabel, trainingFileNames, testData, testLabel, testFileName, varargin)
%TESTANNPARAMS Summary of this function goes here
%   Detailed explanation goes here

    transferFunction = getAdditionalParam( 'transferFunction', varargin, 'tansig' );  % softmax tansig logsig
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    seed = getAdditionalParam( 'seed', varargin, 1 );

    % test training data
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = mlpClassify(...
        trainingData, trainingLabel, trainingData, trainingLabel, trainingFileNames,...
        'seed', seed, 'transferFunction', transferFunction, 'hiddenNodes', hiddenNodes);
    trainingResult = array2table({hiddenNodes accuracy scores mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    % test test data
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = mlpClassify(...
        trainingData, trainingLabel,testData, testLabel, testFileName,...
        'seed', seed, 'transferFunction', transferFunction, 'hiddenNodes', hiddenNodes);
    testResult = array2table({hiddenNodes accuracy scores mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'accuracy', 'scores', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(scores.labels==round(scores.predict_labels));
end

