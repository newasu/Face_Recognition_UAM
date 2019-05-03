% exp2 B find gender by WELM on full DiveFace

clear all

% Settings
numb_run = 1;
numb_cv = 5;
training_sample_percent = 0.75; % percentages of training sample
selected_pose_numb = 1; % number of image used each user

% WELM's parameters
hiddenNodes = 10:10:100;
regularizationC = power(10,-4:1:4);
distFunction = 'euclidean';

% Save path
saveFolderPath = {'Result', 'Exp2', 'Exp2_B'};
filename = [saveFolderPath{end} '_welm'];
save_path = MakeChainFolder(saveFolderPath, 'target_path', pwd);
save_path = [save_path '/' filename];

% %Load data
[diveface_feature, diveface_label] = LoadDiveFaceFull();

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
    trainingDataX_index = table2cell(training_selected_pose_index);
    trainingDataX_index = [trainingDataX_index{:}];
    trainingDataX_index = trainingDataX_index(:);
    trainingDataX = diveface_feature(trainingDataX_index,:);
    trainingDataY = diveface_label.gender(trainingDataX_index);
    trainingFileNames = diveface_label.filename(trainingDataX_index);
    trainingCode = diveface_label.user_code(trainingDataX_index);

    % Test data
    testDataX_index = table2cell(test_selected_pose_index);
    testDataX_index = [testDataX_index{:}];
    testDataX_index = testDataX_index(:);
    testDataX = diveface_feature(testDataX_index,:);
    testDataY = diveface_label.gender(testDataX_index);
    testFileNames = diveface_label.filename(testDataX_index);
    testCode = diveface_label.user_code(testDataX_index);
    
    % Genarate k-fold indices
    [ kFoldIdx, ~ ] = GetKFoldIndices( numb_cv, trainingDataY, random_seed );

    % Find optimal parameter
    [ foldLog, avgFoldLog ] = welmCV(kFoldIdx, trainingDataX, trainingDataY, ...
        trainingFileNames, trainingCode, 'seed', random_seed, 'hiddenNodes', hiddenNodes, ...
        'regularizationC', regularizationC, 'distFunction', distFunction);
    
    % Test model
    [trainingResult, testResult, testCorrectIdx] = TestWELMParams(trainingDataX, ...
        trainingDataY, trainingFileNames, trainingCode, testDataX, testDataY, testFileNames, ...
        'hiddenNodes', table2array(avgFoldLog(1,1)), 'regularizationC',  ...
        table2array(avgFoldLog(1,2)), 'seed', random_seed, 'distFunction', distFunction);
        
    log(random_seed,:) = {foldLog trainingResult testResult};
end

log = cell2table(log, 'variablenames', {'foldLog' 'trainingResult' 'testResult'});
eval([filename ' = log;']);
save(save_path, filename,'-v7.3');


