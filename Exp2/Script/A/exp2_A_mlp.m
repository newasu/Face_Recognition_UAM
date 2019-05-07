% exp1 find gender by MLP

clear all

% Settings
numb_run = 10;
numb_cv = 5;

% WELM's parameters
hiddenNodes = 100;

% Save path
saveFolderPath = {'Result', 'Exp2', 'Exp2_A'};
filename = [saveFolderPath{end} '_mlp'];
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
    trainingFileNames = diveface_label.filename(trainingDataX_index);

    % Test data
    testDataX_index = [test_selected_pose_index{:}];
    testDataX_index = testDataX_index(:);
    testDataX = diveface_feature(testDataX_index,:);
    testDataY = diveface_label.gender(testDataX_index);
    testFileNames = diveface_label.filename(testDataX_index);
    
    % Genarate k-fold indices
    [ kFoldIdx, ~ ] = GetKFoldIndices( numb_cv, trainingDataY, random_seed );

    % Find optimal parameter
    [ foldLog, avgFoldLog ] = mlpCV(kFoldIdx, trainingDataX, trainingFileNames, ...
        double(trainingDataY), 'seed', random_seed, 'hiddenNodes', hiddenNodes, ...
        'transferFunction', 'tansig');
    
    % Test model
    [trainingResult, testResult, testCorrectIdx] = TestMLPParams(trainingDataX, ...
        double(trainingDataY), trainingFileNames, testDataX, double(testDataY), testFileNames, ...
        'hiddenNodes', table2array(avgFoldLog(1,1)), 'transferFunction', 'tansig',  ...
        'seed', random_seed);
        
    log(random_seed,:) = {foldLog trainingResult testResult};
end

log = cell2table(log, 'variablenames', {'foldLog' 'trainingResult' 'testResult'});
eval([filename ' = log;']);
save(save_path, filename,'-v7.3');


