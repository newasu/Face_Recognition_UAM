% test1 find gender by measure euclidean distance

clear all

% Settings
numb_run = 10;

% Save path
saveFolderPath = {'Result', 'Exp2', 'Exp2_sub2'};
filename = [saveFolderPath{end} '_dist'];
save_path = MakeChainFolder(saveFolderPath, 'target_path', pwd);
save_path = [save_path '/' filename];

% %Load data
[diveface_feature, diveface_label] = LoadDiveFace();

for random_seed = 1 : numb_run
    % Split dataset
    [training_id_index, test_id_index, data, user_index] = SplitDataset(diveface_label,...
        'training_sample_percent', 0.9, 'random_seed', random_seed);

    % Select n poses from each user
    [training_selected_pose_index, training_unselected_pose_index] = RandomPickUserPose...
        (training_id_index, diveface_label, user_index, 'selected_pose_numb', 1, 'random_seed', random_seed);
    [test_selected_pose_index, test_unselected_pose_index] = RandomPickUserPose...
        (test_id_index, diveface_label, user_index, 'selected_pose_numb', 1, 'random_seed', random_seed);
        
    % Training data
    trainingDataX_index = [training_selected_pose_index{:}];
    trainingDataX_index = trainingDataX_index(:);
    trainingDataX = diveface_feature(trainingDataX_index,:);
    trainingDataY = diveface_label.gender(trainingDataX_index);

    % Test data
    testDataX_index = [test_selected_pose_index{:}];
    testDataX_index = testDataX_index(:);
    testDataX = diveface_feature(testDataX_index,:);
    testDataY = diveface_label.gender(testDataX_index);
    testFileNames = diveface_label.filename(testDataX_index);
    
    % Class labels
    class_name = categorical(categories(trainingDataY));

    % Predict
    % consider by nearest user
    [~, predict_score] = min(pdist2(testDataX, trainingDataX,'euclidean'), [], 2);
    predictY = trainingDataY(predict_score);
    accuracy_max(random_seed) = (sum(predictY==testDataY)/numel(testDataY)) * 100;
    
    % consider by mean
    predict_score_male = mean(pdist2(testDataX, trainingDataX(trainingDataY == 'male',:),'euclidean'), 2);
    predict_score_female = mean(pdist2(testDataX, trainingDataX(trainingDataY == 'female',:),'euclidean'), 2);
    [~, predict_score] = min([predict_score_female predict_score_male], [], 2);
    predictY = class_name(predict_score);
    accuracy_mean(random_seed) = (sum(predictY==testDataY)/numel(testDataY)) * 100;
end

log = table(accuracy_max', accuracy_mean', 'variablenames', {'accuracy_max', 'accuracy_mean'});
eval([filename ' = log;']);
save(save_path, filename,'-v7.3');


