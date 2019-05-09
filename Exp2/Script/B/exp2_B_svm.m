% exp2 B find gender(Exp2_B) and ethnicity(Exp2_C) by SVM on full DiveFace

clear all

% Settings
numb_run = 1;
numb_cv = 5;
training_sample_percent = 0.75; % percentages of training sample
selected_pose_numb = 1; % number of image used each user

% SVM's parameters
c = power(10,-4:1:4);
kernelType = 'linear';  % linear, rbf
libsvmKernelType = 4; % linear : 0, rbf : 2, precomputed : 4

% Save path
saveFolderPath = {'Result', 'Exp2', 'Exp2_B'};
filename = [saveFolderPath{end} '_svm'];
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
    
    % Genarate k-fold indices
    [ kFoldIdx, ~ ] = GetKFoldIndices( numb_cv, trainingDataY, random_seed );

    % Find optimal parameter for SVM C-SVC by cv
    cvParam = [{c}];
    cvParamName = {'c'};
    cvParam = [{c}];
    cvParamName = {'c'};
    if contains(kernelType, 'rbf')
        cvParam = [cvParam {gamma}]; % put gamma at last for fastest
        cvParamName = [cvParamName 'g'];  
    end
    [ foldLog, avgFoldLog ] = svmCV(kFoldIdx, cvParam, cvParamName,...
        trainingDataX, trainingDataY, trainingFileNames,...
        'kernelType', kernelType, 'libsvmKernelType', libsvmKernelType,...
        'seed', random_seed, 'svmType', 0);

    % Test model
    [trainingResult, testResult, testCorrectIdx] = TestSVMParams(...
        trainingDataX, trainingDataY, trainingFileNames,...
        testDataX, testDataY, testFileNames,...
        avgFoldLog(1,1:numel(cvParam)), cvParamName,...
        'kernelType', kernelType, 'libsvmKernelType', libsvmKernelType,...
        'seed', random_seed, 'svmType', 0);
        
    log(random_seed,:) = {foldLog trainingResult testResult};
end

log = cell2table(log, 'variablenames', {'foldLog' 'trainingResult' 'testResult'});
eval([filename ' = log;']);
save(save_path, filename,'-v7.3');


