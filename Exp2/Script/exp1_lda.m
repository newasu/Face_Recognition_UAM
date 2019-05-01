% test1 find gender by LDA

clear all

% Settings
numb_run = 10;

% Save path
saveFolder = 'Exp1';
filename = 'exp1_lda_result';
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

    % Test data
    testDataX_index = [test_selected_pose_index{:}];
    testDataX_index = testDataX_index(:);
    testDataX = diveface_feature(testDataX_index,:);
    testDataY = diveface_label.gender(testDataX_index);
    testFileNames = diveface_label.filename(testDataX_index);

    % Predict
    [predictY, accuracy, mdl, scores, trainingTime, testTime] = ldaClassify(...
        trainingDataX, trainingDataY, testDataX, testDataY, testFileNames);
    
    log(random_seed) = accuracy;
    
end

eval([filename ' = log;']);
save(save_path, filename,'-v7.3');


