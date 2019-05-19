function [ trainingResult, testResult, testCorrectIdx ] = TestPELM2Params(...
    trainingData_1, trainingData_2, trainingLabel, trainingCode,...
    testData_1, testData_2, testLabel, testCode, varargin)
%TESTPELM2PARAMS Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    regularizationC = getAdditionalParam( 'regularizationC', varargin, 1 );
    combine_rule = getAdditionalParam( 'combine_rule', varargin, 'sum' ); % sum minus multiply distance mean
    seed = getAdditionalParam( 'seed', varargin, 1 );
    select_weight_type = getAdditionalParam( 'select_weight_type', varargin, 'random_select' ); % random_select random_generate
    
    % test training data
    [~, score, mdl, label_mat, trainingTime, testTime] = pelm2Classify(...
        trainingData_1, trainingData_2, trainingLabel, trainingCode,...
        trainingData_1, trainingData_2, trainingLabel, trainingCode,...
        'seed', seed, 'combine_rule', combine_rule,...
        'distFunction', distFunction, 'hiddenNodes', hiddenNodes,...
        'regularizationC', regularizationC, 'select_weight_type', select_weight_type);
    trainingResult = array2table({hiddenNodes regularizationC score.Accuracy label_mat mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'regC', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    
    % test test data
    [~, score, mdl, label_mat, trainingTime, testTime] = pelm2Classify(...
        trainingData_1, trainingData_2, trainingLabel, trainingCode,...
        testData_1, testData_2, testLabel, testCode,...
        'seed', seed, 'combine_rule', combine_rule,...
        'distFunction', distFunction, 'hiddenNodes', hiddenNodes,...
        'regularizationC', regularizationC, 'select_weight_type', select_weight_type);
    testResult = array2table({hiddenNodes regularizationC score.Accuracy label_mat mdl trainingTime testTime},...
        'VariableNames', {'hiddenNodes', 'regC', 'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(label_mat.labels==label_mat.predict_labels);
end

