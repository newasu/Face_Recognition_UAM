% exp2 E find ethnicity by Euclidean on full DiveFace VGG

clear all

% Settings
numb_run = 1;
numb_cv = 5;
training_sample_percent = 0.75; % percentages of training sample
selected_pose_numb = 1; % number of image used each user

% Save path
saveFolderPath = {'Result', 'Exp2', 'Exp2_E'};
filename = [saveFolderPath{end} '_dist'];
save_path = MakeChainFolder(saveFolderPath, 'target_path', pwd);
save_path = [save_path '/' filename];

% %Load data
[diveface_feature, diveface_label] = LoadDiveFaceFull('network_type', 'VGG');

for random_seed = 1 : numb_run
    % Split dataset
    [training_id_index, test_id_index, data, user_index] = SplitDataset(diveface_label,...
        'training_sample_percent', training_sample_percent, 'random_seed', random_seed);

    % Select n poses from each user
    [training_selected_pose_index, training_unselected_pose_index] = RandomPickUserPose...
        (training_id_index, diveface_label, user_index, 'selected_pose_numb',...
        selected_pose_numb, 'random_seed', random_seed);
    [test_selected_pose_index, test_unselected_pose_index] = RandomPickUserPose...
        (test_id_index, diveface_label, user_index, 'selected_pose_numb',...
        selected_pose_numb, 'random_seed', random_seed);

    % Training data
    trainingData_index = table2cell(training_selected_pose_index);
    trainingData_index = [trainingData_index{:}];
    trainingData_index = trainingData_index(:);
    trainingDataX = diveface_feature(trainingData_index,:);
%     trainingDataY = diveface_label.gender(trainingData_index);
    trainingDataY = diveface_label.ethnicity(trainingData_index);
    trainingFileNames = diveface_label.filename(trainingData_index);
    training_data_id = diveface_label.data_id(trainingData_index);

    % Test data
    testData_index = table2cell(test_selected_pose_index);
    testData_index = [testData_index{:}];
    testData_index = testData_index(:);
    testDataX = diveface_feature(testData_index,:);
%     testDataY = diveface_label.gender(testData_index);
    testDataY = diveface_label.ethnicity(testData_index);
    testFileNames = diveface_label.filename(testData_index);
    test_data_id = diveface_label.data_id(testData_index);
    
    % Class labels
    class_name = categorical(categories(trainingDataY));

    % Predict
    tic
    dist_matrix = pdist2(testDataX, trainingDataX,'euclidean');
    trainingTime = toc;
    
    % Consider by nearest user
    tic;
    [~, predict_score] = min(dist_matrix, [], 2);
    testTime = toc;
    predictY = trainingDataY(predict_score);
%     score = sum(predictY==testDataY)/numel(testDataY);
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    
    % Save max method
    label_mat = table(testFileNames, testDataY, predictY,...
        'VariableNames', {'filenames' 'labels', 'predict_labels'});
    testResult = array2table({  score.F1_score label_mat training_data_id trainingTime testTime},...
        'VariableNames', {'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    log_near(random_seed, :) = {[] [] testResult};
    
    % Consider by mean
    predict_score = [];
    tic;
    for i = 1 : numel(class_name)
        predict_score(:,i) = mean(dist_matrix(:, (trainingDataY == class_name(i))), 2);
    end
    trainingTime = trainingTime + toc;
    tic;
    [~, predict_score] = min(predict_score, [], 2);
    testTime = toc;
    predictY = class_name(predict_score);
% 	score(random_seed) = sum(predictY==testDataY)/numel(testDataY);
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    
    % Save mean method
    label_mat = table(testFileNames, testDataY, predictY,...
        'VariableNames', {'filenames' 'labels', 'predict_labels'});
    testResult = array2table({  score.F1_score label_mat training_data_id trainingTime testTime},...
        'VariableNames', {'score', 'label_mat', 'model', 'trainingTime', 'testTime'});
    log_mean(random_seed, :) = {[] [] testResult};
        
end

log_near = cell2table(log_near, 'variablenames', {'foldLog' 'trainingResult' 'testResult'});
eval([filename '_near = log_near;']);
save([save_path '_near'], [filename '_near'],'-v7.3');

log_mean = cell2table(log_mean, 'variablenames', {'foldLog' 'trainingResult' 'testResult'});
eval([filename '_mean = log_mean;']);
save([save_path '_mean'], [filename '_mean'],'-v7.3');


