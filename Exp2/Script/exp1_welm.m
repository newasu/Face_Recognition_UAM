% exp1 find gender by LDA

clear all

% Settings
numb_run = 10;
numb_cv = 5;

% WELM's parameters
hiddenNodes = 10:10:100;
regularizationC = power(10,-4:1:4);
distFunction = 'euclidean';

% Save path
saveFolder = 'Exp1';
filename = 'exp1_welm_result';
save_path = [pwd '/Result/' saveFolder '/' filename];

% Make folder if not exist
if ~exist([pwd '/Result' ],'dir')
    mkdir([pwd '/Result' ]);
end
if ~exist([pwd '/Result/' saveFolder ],'dir')
    mkdir([pwd '/Result/' saveFolder ]);
end

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
    [ foldLog, avgFoldLog ] = welmCV(kFoldIdx, trainingDataX, trainingFileNames, ...
        double(trainingDataY), 'seed', random_seed, 'hiddenNodes', hiddenNodes, ...
        'regularizationC', regularizationC, 'distFunction', distFunction);
    
    % Test model
    [trainingResult, testResult, testCorrectIdx] = TestWELMParams(trainingDataX, ...
        double(trainingDataY), trainingFileNames, testDataX, double(testDataY), testFileNames, ...
        'hiddenNodes', table2array(avgFoldLog(1,1)), 'regularizationC',  ...
        table2array(avgFoldLog(1,2)), 'seed', random_seed, 'distFunction', distFunction);
        
    log(random_seed,:) = {foldLog trainingResult testResult};
end

log = cell2table(log, 'variablenames', {'foldLog' 'trainingResult' 'testResult'});
eval([filename ' = log;']);
save(save_path, filename,'-v7.3');


