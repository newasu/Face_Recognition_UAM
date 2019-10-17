function [foldLog, avgFoldLog] = dist_parallelCV_new(foldIdx, data_1, data_2, labels, data_code, varargin)
%DIST_PARALLELCV Summary of this function goes here
%   Detailed explanation goes here

    distFunction = getAdditionalParam( 'distFunction', varargin, 'euclidean' );  % euclidean cosine
    seed = getAdditionalParam( 'seed', varargin, 1 );

    countingRound = 0;
    noFold = size(foldIdx,1);
    foldLog = [];
    for fold = 1 : noFold
        trainingData_1 = data_1;
        trainingData_2 = data_2;
        trainingLabel = labels;
        trainingFileNames = data_code;
        
        % train 4 out of 5 part
        trainingData_1(foldIdx(fold,:),:) = [];
        trainingData_2(foldIdx(fold,:),:) = [];
        trainingLabel(foldIdx(fold,:),:) = [];
        trainingFileNames(foldIdx(fold,:),:) = [];
        
        % test the rest part
        testData_1 = data_1(foldIdx(fold,:),:);
        testData_2 = data_2(foldIdx(fold,:),:);
        testLabel = labels(foldIdx(fold,:),:);
        testFileNames = data_code(foldIdx(fold,:),:);
        
        class_name = categorical(unique(trainingLabel));
        class_same_idx = find(class_name == 'same');
        class_different_idx = find(class_name == 'different');
        
        % Train
        tic;
        
        XX = sum(trainingData_1.^2, 2);
        XY = sum(trainingData_1 .* trainingData_2,2);
        YY = sum(trainingData_2.^2, 2);
        train_dist_score = sqrt(XX - (2*XY) + YY);
        clear XX XY YY
        
        methodParam = (floor(min(train_dist_score)*10)/10):0.1:(ceil(max(train_dist_score)*10)/10);

        dist_confusion = [];
        for i = 1 : numel(methodParam)
            same_idx = train_dist_score <= methodParam(i);
            predict_labels = repmat(class_name(class_different_idx), numel(trainingLabel), 1);
            predict_labels(same_idx) = class_name(class_same_idx);
            c = confusionmat(trainingLabel, predict_labels);
            dist_confusion = [dist_confusion; methodParam(i), c(1,end), c(end,1)];
        end
        
        dist_confusion(:,2) = dist_confusion(:,2)/max(dist_confusion(:,2));
        dist_confusion(:,3) = dist_confusion(:,3)/max(dist_confusion(:,3));
%         plot(dist_confusion(:,1),dist_confusion(:,2),dist_confusion(:,1),dist_confusion(:,3))
        
        crossing_point = InterX([dist_confusion(:,1)'; dist_confusion(:,2)'],...
            [dist_confusion(:,1)'; dist_confusion(:,3)']);
        crossing_point = crossing_point(1);
        trainingTime = toc;
        
        % Test
        tic;
        
        XX = sum(testData_1.^2, 2);
        XY = sum(testData_1 .* testData_2,2);
        YY = sum(testData_2.^2, 2);
        test_dist_score = sqrt(XX - (2*XY) + YY);
        clear XX XY YY

        same_idx = test_dist_score <= crossing_point;
        predict_labels = repmat(class_name(class_different_idx), numel(testLabel), 1);
        predict_labels(same_idx) = class_name(class_same_idx);
        testTime = toc;
        
        predict_score = test_dist_score;
        [~,score,~] = my_confusion.getMatrix(double(testLabel),double(predict_labels),0);
        label_mat = table(testFileNames, testLabel, predict_labels, predict_score, ...
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
        
        foldLog = [foldLog; crossing_point fold score.FMR_0d01 score.AUC score.EER score.FMR_0d1 {label_mat} trainingTime testTime];
        
        % Count Progress Bar
        countingRound = countingRound + 1;
        disp([ 'Seed: '  num2str(seed) ', Finished: ' num2str(fold) '/' num2str(noFold)]);
        
    end
    
    numbCVParam = 1;
    foldLog = sortrows(foldLog,[(numbCVParam+1)]);
    foldLog = array2table(foldLog, 'VariableNames', {'threshold', 'fold', ...
        'score_FMR_0d01', 'score_AUC', 'score_EER', 'score_FMR_0d1', ...
        'label_mat', 'trainingTime', 'testTime'});
    
    % Average 5 folds into 1
    %     avgFoldLog = cell2mat(table2array(foldLog(:, [(1:numbCVParam) 2+numbCVParam 4+numbCVParam 5+numbCVParam ])));
    avgFoldLog = cell2mat(table2array(foldLog(:, (1:numbCVParam))));
    avgFoldLog = [avgFoldLog cell2mat(table2array(foldLog(:, {'score_FMR_0d01', 'score_AUC', 'score_EER', 'score_FMR_0d1', 'trainingTime', 'testTime'})))];
    avgFoldLog = reshape(avgFoldLog',[(size(avgFoldLog,2)), noFold, (size(avgFoldLog,1)/noFold)]);
    avgFoldLog = sum(avgFoldLog,2)/noFold;
    avgFoldLog = reshape(avgFoldLog,[size(avgFoldLog,1) size(avgFoldLog,3)])';
    avgFoldLog = array2table(sortrows(avgFoldLog,[(numbCVParam+1) 1]), ...
        'VariableNames', {'threshold', 'score_FMR_0d01', 'score_AUC', 'score_EER', 'score_FMR_0d1', 'trainingTime', 'testTime'});
    
end

