% exp2 D find gender by LDA on full DiveFace VGG
clear all

% Settings
numb_run = 1;
numb_cv = 5;
training_sample_percent = 0.75; % percentages of training sample
selected_pose_numb = 1; % number of image used each user

% Save path
saveFolderPath = {'Result', 'Exp2', 'Exp2_D'};
filename = [saveFolderPath{end} '_lda'];
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
    trainingDataY = diveface_label.gender(trainingData_index);
%     trainingDataY = diveface_label.ethnicity(trainingData_index);
    trainingFileNames = diveface_label.filename(trainingData_index);
    training_data_id = diveface_label.data_id(trainingData_index);

    % Test data
    testData_index = table2cell(test_selected_pose_index);
    testData_index = [testData_index{:}];
    testData_index = testData_index(:);
    testDataX = diveface_feature(testData_index,:);
    testDataY = diveface_label.gender(testData_index);
%     testDataY = diveface_label.ethnicity(testData_index);
    testFileNames = diveface_label.filename(testData_index);
    test_data_id = diveface_label.data_id(testData_index);
    
    % Find optimal parameter
    [ foldLog, avgFoldLog ] = ldaCV( trainingDataX, trainingDataY, trainingFileNames);
    
    % Test model
    [ trainingResult, testResult, testCorrectIdx ] = TestLDAParams(...
        trainingDataX, trainingDataY, trainingFileNames,...
        testDataX, testDataY, testFileNames, 'kernelType', 'linear');
    
        
    log(random_seed,:) = {foldLog trainingResult testResult};
end

log = cell2table(log, 'variablenames', {'foldLog' 'trainingResult' 'testResult'});
eval([filename ' = log;']);
save(save_path, filename,'-v7.3');


