function [ trainingResult, testResult, testCorrectIdx ] = TestANNParams(...
    trainingData, trainingLabel, trainingFileNames, testData, testLabel, testFileName, varargin)
%TESTANNPARAMS Summary of this function goes here
%   Detailed explanation goes here

%     transferFunction = getAdditionalParam( 'transferFunction', varargin, 'tansig' );  % softmax tansig logsig
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    seed = getAdditionalParam( 'seed', varargin, 1 );

    % test training data
    [predictY, score, mdl, label_mat, trainingTime, testTime] = mlpClassify(...
        trainingData, trainingLabel, trainingData, trainingLabel, trainingFileNames,...
        'seed', seed, 'hiddenNodes', hiddenNodes);
    trainingResult = array2table({hiddenNodes score label_mat mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    % test test data
    [predictY, score, mdl, label_mat, trainingTime, testTime] = mlpClassify(...
        trainingData, trainingLabel,testData, testLabel, testFileName,...
        'seed', seed, 'hiddenNodes', hiddenNodes);
    testResult = array2table({hiddenNodes score label_mat mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(label_mat.labels==label_mat.predict_labels);
end

