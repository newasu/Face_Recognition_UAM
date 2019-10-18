% exp4 B classify Is that same person by euclidean distance

clear all

% Settings

experiment_mode = '';
experiment_name = 'Exp4';
sub_experiment_name = 'B';
method_name = 'dist';
do_experiment = [1 2 3 4 5 6 7 8 9 10]; % run experiment number

number_subdataset = 10;
numb_cv = 5; % cv for optimal parameters

% Distance's parameter
distFunction = 'euclidean'; % euclidean cosine

% Save path
default_data_store_path = pwd;
idcs = strfind(pwd,filesep);
default_data_store_path = [default_data_store_path(1:idcs(end)-1) ...
    filesep 'Face_Recognition_UAM_data_store'];
saveFolderPath = {'Result', experiment_name, [experiment_name '_' sub_experiment_name]};
filename = [saveFolderPath{end} '_' method_name];
save_path = MakeChainFolder(saveFolderPath, 'target_path', default_data_store_path);
paired_list_save_path = MakeChainFolder({'Other', experiment_name}, 'target_path', default_data_store_path);
clear idcs default_data_store_path saveFolderPath

% %Load data
[diveface_feature, diveface_label] = LoadDiveFaceFull();
diveface_data_id = diveface_label.data_id;

% Remove user containing less than contain_image
disp('Removing error users..');
user_contain_image = 3;
id = unique(diveface_label.id);
for i = 1 : numel(id)
    temp = find(diveface_label.id == id(i));
    if numel(temp) < user_contain_image
        disp('Deleting user data_id: ');
        num2str(diveface_data_id(temp))
        diveface_label(temp,:) = [];
        diveface_feature(temp,:) = [];
        diveface_data_id(temp) = [];
    end
end
clear contain_image id i temp
disp('Removed error users');

% Generate training/test set
[subdataset, subdataset_label] = SplitSubdataset(diveface_label.id, ...
    {diveface_label.gender, diveface_label.ethnicity}, ...
    'number_sub_dataset', number_subdataset);
concat_subdataset_label = strcat(subdataset_label(:,1), '_', subdataset_label(:,2));

% Test
if strcmp(experiment_mode, 'test')
    subdataset = cellfun(@(x) x(1:5,:), subdataset, 'UniformOutput', false);
    warning('RUNNING IN TEST MODE!!');
end

for experiment_round = do_experiment
    
    % Prepare data
    disp('Preparing data..');
    
    % Assign training/test set
    training_set_sample_idx = [];
    test_set_sample_idx = [];
    for i = 1 : number_subdataset % Train one and test nine
        if experiment_round == i % training set
            training_set_sample_idx = subdataset{i};
        else % test set
            test_set_sample_idx = [test_set_sample_idx; subdataset{i}];
        end
    end

    % Load training paired sample list
    temp_paired_list_save_path = [paired_list_save_path filesep ...
        'training_paired_label_' num2str(experiment_round)];
    if exist([temp_paired_list_save_path '.mat'], 'file') && ~strcmp(experiment_mode, 'test')
        disp('Loading training_paired_label..');
        load(temp_paired_list_save_path);
    else
        disp('Loading training_paired_label..');
        training_paired_label = PairSampleClassesEqually(...
            training_set_sample_idx, subdataset_label, diveface_label,...
            'random_seed', experiment_round);
%         save(temp_paired_list_save_path, 'training_paired_label', '-v7.3');
    end

    % Load test paired sample list
    temp_paired_list_save_path = [paired_list_save_path filesep ...
        'test_paired_label_' num2str(experiment_round)];
    if exist([temp_paired_list_save_path '.mat'], 'file') && ~strcmp(experiment_mode, 'test')
        disp('Loading test_paired_label..');
        load(temp_paired_list_save_path);
    else
        disp('Generateing test_paired_label..');
        test_paired_label = PairSampleClassesEqually(...
            test_set_sample_idx, subdataset_label, diveface_label,...
            'random_seed', experiment_round);
%         save(temp_paired_list_save_path, 'test_paired_label', '-v7.3');
    end

    % Subsampling
    disp('Subsampling data..');
    % Sort
    training_paired_label = sortrows(training_paired_label,[2 -1 3 12]);
    test_paired_label = sortrows(test_paired_label,[2 -1 3 12]);

    % Remove some class of different data in training_paired_label for subsampling
    temp = training_paired_label.paired_label == 'different';
    temp_temp = (1 : size(training_paired_label,1))';
    temp_temp = temp_temp(temp);
    temp = training_paired_label.master_data_id(temp);
    temp = reshape(temp,[numel(concat_subdataset_label) (numel(temp)/numel(concat_subdataset_label))]);
    temp_temp  = reshape(temp_temp,[numel(concat_subdataset_label) (numel(temp_temp)/numel(concat_subdataset_label))]);
    temp_temp_temp = temp_temp;
    temp = [];
    for i = 1 : (size(temp_temp_temp,2) / (numel(concat_subdataset_label)/2))
        for j = 1 : numel(concat_subdataset_label)/2
            temp = [temp; temp_temp(((j*2)-1):(j*2), j)];
        end
        temp_temp(:,1:numel(concat_subdataset_label)/2) = [];
    end
    training_paired_label = [training_paired_label(training_paired_label.paired_label == 'same', :); ...
        training_paired_label(temp,:)];
    clear temp temp_temp temp_temp_temp

    % Remove some class of different data in test_paired_label for subsampling
    temp = test_paired_label.paired_label == 'different';
    temp_temp = (1 : size(test_paired_label,1))';
    temp_temp = temp_temp(temp);
    temp = test_paired_label.master_data_id(temp);
    temp = reshape(temp,[numel(concat_subdataset_label) (numel(temp)/numel(concat_subdataset_label))]);
    temp_temp  = reshape(temp_temp,[numel(concat_subdataset_label) (numel(temp_temp)/numel(concat_subdataset_label))]);
    temp_temp_temp = temp_temp;
    temp = [];
    for i = 1 : (size(temp_temp_temp,2) / (numel(concat_subdataset_label)/2))
        for j = 1 : numel(concat_subdataset_label)/2
            temp = [temp; temp_temp(((j*2)-1):(j*2), j)];
        end
        temp_temp(:,1:numel(concat_subdataset_label)/2) = [];
    end
    test_paired_label = [test_paired_label(test_paired_label.paired_label == 'same', :); ...
        test_paired_label(temp,:)];
    clear temp temp_temp temp_temp_temp i j
    % Sort
    training_paired_label = sortrows(training_paired_label,[2 -1 3 12]);
    test_paired_label = sortrows(test_paired_label,[2 -1 3 12]);

    % Check subsampling correction
    disp('Checking subsampling correction..');
    
    % Check correction of users's subsampling in training_paired_label
    temp = strcat(cellstr(training_paired_label.paired_gender), '_', cellstr(training_paired_label.paired_ethnicity));
    temp = temp(training_paired_label.paired_label=='different');
    temp = reshape(temp, [numel(concat_subdataset_label) (numel(temp)/numel(concat_subdataset_label))]);
    temp = mat2cell(temp', ones(1,size(temp,2)));
    temp = cell2mat(cellfun(@(x) sum(ismember(x,concat_subdataset_label)), temp, 'UniformOutput', false));
    if sum(temp ~= numel(concat_subdataset_label)) > 0
        error('Error proportion of class labels not balance')
    end

    % Check correction of users's subsampling in test_paired_label
    temp = strcat(cellstr(test_paired_label.paired_gender), '_', cellstr(test_paired_label.paired_ethnicity));
    temp = temp(test_paired_label.paired_label=='different');
    temp = reshape(temp, [numel(concat_subdataset_label) (numel(temp)/numel(concat_subdataset_label))]);
    temp = mat2cell(temp', ones(1,size(temp,2)));
    temp = cell2mat(cellfun(@(x) sum(ismember(x,concat_subdataset_label)), temp, 'UniformOutput', false));
    if sum(temp ~= numel(concat_subdataset_label)) > 0
        error('Error proportion of class labels not balance')
    end
    
    clear temp
    
    % Remove unused category
    disp('Removing unused category..');
    training_paired_label.paired_label = removecats(training_paired_label.paired_label);
    training_paired_label.master_gender = removecats(training_paired_label.master_gender);
    training_paired_label.master_ethnicity = removecats(training_paired_label.master_ethnicity);
    training_paired_label.paired_gender = removecats(training_paired_label.paired_gender);
    training_paired_label.paired_ethnicity = removecats(training_paired_label.paired_ethnicity);
    test_paired_label.master_gender = removecats(test_paired_label.master_gender);
    test_paired_label.master_ethnicity = removecats(test_paired_label.master_ethnicity);
    test_paired_label.paired_gender = removecats(test_paired_label.paired_gender);
    test_paired_label.paired_ethnicity = removecats(test_paired_label.paired_ethnicity);
    
    % Genarate k-fold indices
    [kFoldIdx_id, kFoldIdx_data_id]= GetKFoldIdxSepID( numb_cv, training_paired_label.master_id, experiment_round );

    % Assign data
    disp('Assigning training/test data..');
    
    % Training data
    trainingDataX_1 = cell2mat(arrayfun(@(x) find(x==diveface_data_id), ...
        training_paired_label.master_data_id, 'UniformOutput', false));
    trainingDataX_1 = diveface_feature(trainingDataX_1,:);
    trainingDataX_2 = cell2mat(arrayfun(@(x) find(x==diveface_data_id), ...
        training_paired_label.paired_data_id, 'UniformOutput', false));
    trainingDataX_2 = diveface_feature(trainingDataX_2,:);
    training_data_id = [training_paired_label.master_data_id training_paired_label.paired_data_id];
    trainingDataY = training_paired_label.paired_label;

%     % Test data
    testDataX_1 = cell2mat(arrayfun(@(x) find(x==diveface_data_id), ...
        test_paired_label.master_data_id, 'UniformOutput', false));
    testDataX_1 = diveface_feature(testDataX_1,:);
    testDataX_2 = cell2mat(arrayfun(@(x) find(x==diveface_data_id), ...
        test_paired_label.paired_data_id, 'UniformOutput', false));
    testDataX_2 = diveface_feature(testDataX_2,:);
    test_data_id = [test_paired_label.master_data_id test_paired_label.paired_data_id];
    testDataY = test_paired_label.paired_label;
    
    disp('Finished preparing data');

    % Run experiment
    disp('Running experiment..');
    
    % stopwatch
    running_time = tic;
    
    % Find optimal parameter
    disp(['Running ' num2str(numb_cv) '-fold cross validation..']);
    [foldLog, avgFoldLog] = dist_parallelCV_new(kFoldIdx_data_id, trainingDataX_1, trainingDataX_2, ...
    	trainingDataY, training_data_id, 'distFunction', distFunction, 'seed', experiment_round);
    
    % Test model
    disp('Testing model..');
    [trainingResult, testResult, ~] = TestDist_parallelParams( ...
        trainingDataX_1, trainingDataX_2, trainingDataY, training_data_id, ...
        testDataX_1, testDataX_2, testDataY, test_data_id, ...
        'distFunction', distFunction);
    
    % Save
    if ~strcmp(experiment_mode, 'test')
        disp('Saving..');
        data_log = table({foldLog}, {trainingResult}, {testResult}, ...
            'variablenames', {'foldLog' 'trainingResult' 'testResult'});
        my_filename = [filename '_' distFunction '_' num2str(experiment_round)];
        my_save_path = [save_path filesep my_filename];
        eval([my_filename ' = data_log;']);
        save(my_save_path, my_filename,'-v7.3');
        eval(['clear ' my_filename])
    end

    clear foldLog avgFoldLog trainingResult testResult data_log
    
    % stopwatch
    running_time = toc(running_time);

    disp(['Finished experiment ' num2str(experiment_round) ' from ' method_name ' ' distFunction '.']);
    MailNotify('Subject', ['Experiment ' my_filename ' finished'], ...
        'Message', ['Experiment ' my_filename ' has been finished at ' ...
        char(datetime('now','TimeZone','+07:00')) '. (' num2str(running_time) ' seconds)']);
    
    clear training_paired_label test_paired_label
end

MailNotify('Subject', 'All experiments have been done');
