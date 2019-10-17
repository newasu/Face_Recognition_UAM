function [ trainingResult, testResult, testCorrectIdx ] = TestDist_parallelParams(...
    trainingData_1, trainingData_2, trainingLabel, trainingCode,...
    testData_1, testData_2, testLabel, testCode, varargin)
%TESTDIST_PARALLELPARAMS Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    threshold = getAdditionalParam( 'threshold', varargin, 1 );
    
    class_name = categorical(unique(trainingLabel));
    class_same_idx = find(class_name == 'same');
    class_different_idx = find(class_name == 'different');
    
    % Train
    trainingTime = 0;
    tic;
    
    XX = sum(trainingData_1.^2, 2);
    XY = sum(trainingData_1 .* trainingData_2,2);
    YY = sum(trainingData_2.^2, 2);
    train_dist_score = sqrt(XX - (2*XY) + YY);
    clear XX XY YY

    same_idx = train_dist_score <= threshold;
    predict_labels = repmat(class_name(class_different_idx), numel(trainingLabel), 1);
    predict_labels(same_idx) = class_name(class_same_idx);
    testTime = toc;
    
    predict_score = train_dist_score;
    [~,score,~] = my_confusion.getMatrix(double(trainingLabel),double(predict_labels),0);
    label_mat = table(trainingCode, trainingLabel, predict_labels, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
    predict_score(predict_score == inf) = 0;
    
    % AUC
    auc = [];
    for ii = 1 : numel(class_name)
        temp = trainingLabel==class_name(ii);
        if class_name(ii) == class_name(class_same_idx)
            [~,~,~,auc(ii)] = perfcurve(temp, -predict_score,1);
        else
            [~,~,~,auc(ii)] = perfcurve(temp, predict_score,1);
        end
    end
    score.AUC = sum(auc)/numel(class_name);

    % biometric score
    [biometric_perf_threshold, biometric_perf_mat] = exp3_report_biometric_perf(...
        trainingLabel, predict_labels, predict_score, class_name(class_same_idx), ...
        'positive_class_score_order', 'descend');
    score.EER = biometric_perf_mat.EER;
    score.FMR_0d1 = biometric_perf_mat.FMR_0d1;
    score.FMR_0d01 = biometric_perf_mat.FMR_0d01;
    
    trainingResult = array2table({threshold score.FMR_0d01 score.AUC score.EER score.FMR_0d1 label_mat trainingTime testTime},...
        'VariableNames', {'threshold', 'score_FMR_0d01', 'score_AUC', 'score_EER', 'score_FMR_0d1', 'label_mat', 'trainingTime', 'testTime'});

    % Test
    trainingTime = 0;
    tic;

    XX = sum(testData_1.^2, 2);
    XY = sum(testData_1 .* testData_2,2);
    YY = sum(testData_2.^2, 2);
    test_dist_score = sqrt(XX - (2*XY) + YY);
    clear XX XY YY
    
    same_idx = test_dist_score <= threshold;
    predict_labels = repmat(class_name(class_different_idx), numel(testLabel), 1);
    predict_labels(same_idx) = class_name(class_same_idx);
    testTime = toc;
    
    predict_score = test_dist_score;
    [~,score,~] = my_confusion.getMatrix(double(testLabel),double(predict_labels),0);
    label_mat = table(testCode, testLabel, predict_labels, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
    predict_score(predict_score == inf) = 0;
    
    % AUC
    auc = [];
    for ii = 1 : numel(class_name)
        temp = testLabel==class_name(ii);
        if class_name(ii) == class_name(class_same_idx)
            [~,~,~,auc(ii)] = perfcurve(temp, -predict_score,1);
        else
            [~,~,~,auc(ii)] = perfcurve(temp, predict_score,1);
        end
    end
    score.AUC = sum(auc)/numel(class_name);

    % biometric score
    [biometric_perf_threshold, biometric_perf_mat] = exp3_report_biometric_perf(...
        testLabel, predict_labels, predict_score, class_name(class_same_idx), ...
        'positive_class_score_order', 'descend');
    score.EER = biometric_perf_mat.EER;
    score.FMR_0d1 = biometric_perf_mat.FMR_0d1;
    score.FMR_0d01 = biometric_perf_mat.FMR_0d01;
    
    testResult = array2table({threshold score.FMR_0d01 score.AUC score.EER score.FMR_0d1 label_mat trainingTime testTime},...
        'VariableNames', {'threshold', 'score_FMR_0d01', 'score_AUC', 'score_EER', 'score_FMR_0d1', 'label_mat', 'trainingTime', 'testTime'});
    
    testCorrectIdx = find(label_mat.labels==label_mat.predict_labels);
    
end

