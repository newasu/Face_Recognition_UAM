function [foldLog, avgFoldLog] = dist_parallelCV(foldIdx, data_1, data_2, labels, data_code, varargin)
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
        
        % Train
        tic;
        train_dist_score = [];
        for ii = 1 : numel(trainingLabel)
            train_dist_score(ii) = pdist2(trainingData_1(ii,:), trainingData_2(ii,:), distFunction);
        end
        
        methodParam = (floor(min(train_dist_score)*10)/10):0.1:(ceil(max(train_dist_score)*10)/10);

        dist_confusion = [];
        for i = 1 : numel(methodParam)
            same_idx = find(train_dist_score <= methodParam(i));
            predict_labels = repmat(class_name(1), numel(trainingLabel), 1);
            predict_labels(same_idx) = class_name(2);
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
        test_dist_score = [];
        for ii = 1 : numel(testLabel)
            test_dist_score(ii) = pdist2(testData_1(ii,:), testData_2(ii,:), distFunction);
        end
        same_idx = find(test_dist_score <= crossing_point);
        predict_labels = repmat(class_name(1), numel(testLabel), 1);
        predict_labels(same_idx) = class_name(2);
        testTime = toc;
        predict_score = test_dist_score';
        [~,score,~] = my_confusion.getMatrix(double(testLabel),double(predict_labels),0);
        label_mat = table(testFileNames, testLabel, predict_labels, predict_score, ...
            'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
        
        foldLog = [foldLog; crossing_point fold score.Accuracy {label_mat} trainingTime testTime];
        
        % Count Progress Bar
        countingRound = countingRound + 1;
        disp([ 'Seed: '  num2str(seed) ', Finished: ' num2str(fold) '/' num2str(noFold)]);
        
    end
    
    numbCVParam = 1;
    foldLog = sortrows(foldLog,[(numbCVParam+1)]);
    foldLog = array2table(foldLog, 'VariableNames', {'threshold', 'fold', 'score', 'label_mat', 'trainingTime', 'testTime'});
    
    % Average 5 folds into 1
    avgFoldLog = cell2mat(table2array(foldLog(:, [(1:numbCVParam) 2+numbCVParam 4+numbCVParam 5+numbCVParam ])));
    avgFoldLog = reshape(avgFoldLog',[(size(avgFoldLog,2)), noFold, (size(avgFoldLog,1)/noFold)]);
    avgFoldLog = sum(avgFoldLog,2)/noFold;
    avgFoldLog = reshape(avgFoldLog,[size(avgFoldLog,1) size(avgFoldLog,3)])';
    avgFoldLog = array2table(sortrows(avgFoldLog,[-(numbCVParam+1) 1]), ...
        'VariableNames', {'threshold', 'score', 'trainingTime', 'testTime'});
    
end

