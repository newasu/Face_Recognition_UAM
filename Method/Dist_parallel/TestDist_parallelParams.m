function [ trainingResult, testResult, testCorrectIdx ] = TestDist_parallelParams(...
    trainingData_1, trainingData_2, trainingLabel, trainingCode,...
    testData_1, testData_2, testLabel, testCode, varargin)
%TESTDIST_PARALLELPARAMS Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    threshold = getAdditionalParam( 'threshold', varargin, 1 );
    
    class_name = categorical(unique(trainingLabel));
    
    % Train
    trainingTime = 0;
    tic;
    train_dist_score = [];
    for ii = 1 : numel(trainingLabel)
        train_dist_score(ii) = pdist2(trainingData_1(ii,:), trainingData_2(ii,:), distFunction);
    end

    same_idx = find(train_dist_score <= threshold);
    predict_labels = repmat(class_name(1), numel(trainingLabel), 1);
    predict_labels(same_idx) = class_name(2);
    testTime = toc;
    
    predict_score = train_dist_score';
    
    [~,score,~] = my_confusion.getMatrix(double(trainingLabel),double(predict_labels),0);
    label_mat = table(trainingCode, trainingLabel, predict_labels, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
    
    trainingResult = array2table({threshold score.Accuracy label_mat trainingTime testTime},...
        'VariableNames', {'threshold', 'score', 'label_mat', 'trainingTime', 'testTime'});

    % Test
    trainingTime = 0;
    tic;
    test_dist_score = [];
    for ii = 1 : numel(testLabel)
        test_dist_score(ii) = pdist2(testData_1(ii,:), testData_2(ii,:), distFunction);
    end
    same_idx = find(test_dist_score <= threshold);
    predict_labels = repmat(class_name(1), numel(testLabel), 1);
    predict_labels(same_idx) = class_name(2);
    testTime = toc;
    
    predict_score = test_dist_score';
    
    [~,score,~] = my_confusion.getMatrix(double(testLabel),double(predict_labels),0);
    label_mat = table(testCode, testLabel, predict_labels, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
    
    testResult = array2table({threshold score.Accuracy label_mat trainingTime testTime},...
        'VariableNames', {'threshold', 'score', 'label_mat', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(label_mat.labels==label_mat.predict_labels);
    
end

