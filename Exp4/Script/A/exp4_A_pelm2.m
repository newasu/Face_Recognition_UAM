% exp4 A classify Is that same person by PELM2

clear all

% Settings
numb_run = 1;
numb_cv = 2;
training_sample_percent = 0.001; % percentages of training sample
selected_pose_numb = 3; % number of image used each user
number_comparison = 1;
do_balance_class = 1;

% PELM's parameters
hiddenNodes = 100;
regularizationC = power(10,-6:1:6);
select_weight_type = 'random_select'; % random_select random_generate
distFunction = 'euclidean';
combine_rule = {'sum', 'multiply', 'distance', 'mean', 'linear', 'euclidean'};
% sum multiply distance mean linear euclidean

% Save path
default_data_store_path = pwd;
idcs = strfind(pwd,filesep);
default_data_store_path = [default_data_store_path(1:idcs(end)-1) ...
    filesep 'Face_Recognition_UAM_data_store'];
saveFolderPath = {'Result', 'Exp4', 'Exp4_A'};
filename = [saveFolderPath{end} '_pelm2'];
save_path = MakeChainFolder(saveFolderPath, 'target_path', default_data_store_path);

% %Load data
[diveface_feature, diveface_label] = LoadDiveFaceFull();

% Remove user containing less than contain_image
contain_image = 3;
id = unique(diveface_label.id);
for i = 1 : numel(id)
    temp = find(diveface_label.id == id(i));
    if numel(temp) < contain_image
        diveface_label(temp,:) = [];
    end
end
clear contain_image id i temp

for random_seed = 1 : numb_run
    % Split dataset
    [training_id_index, test_id_index, data, user_index] = SplitDataset(diveface_label,...
        'training_sample_percent', training_sample_percent, 'random_seed', random_seed);
    
    [training_pair_list, training_label_pair_list] = PairDataEachClass(...
        training_id_index, diveface_label,...
        'random_seed', random_seed, 'number_comparison', number_comparison, ...
        'selected_pose_numb', selected_pose_numb);
    
    % Subsampling to balance training set
    if do_balance_class
        [training_pair_list, training_label_pair_list] = BalanceClasses(...
            training_pair_list, training_label_pair_list, 'random_seed', random_seed);
    end

    % Training data
    trainingDataX_1 = diveface_feature(training_pair_list(:,1),:);
    trainingDataX_2 = diveface_feature(training_pair_list(:,2),:);
    training_data_id = training_pair_list;
    trainingDataY = training_label_pair_list;
    
    % Genarate k-fold indices
    [ kFoldIdx, ~ ] = GetKFoldIndices( numb_cv, trainingDataY, random_seed );

    for i = 1 : numel(combine_rule)
        % Find optimal parameter
        [ foldLog, avgFoldLog ] = pelm2CV(kFoldIdx, ...
            trainingDataX_1, trainingDataX_2, trainingDataY, training_data_id, ...
            'seed', random_seed, 'hiddenNodes', hiddenNodes, 'regularizationC', ...
            regularizationC, 'distFunction', distFunction, 'combine_rule', combine_rule{i}, ...
            'select_weight_type', select_weight_type);

        data_log(random_seed,:) = {foldLog avgFoldLog};
        data_log = cell2table(data_log, 'variablenames', {'foldLog' 'avgFoldLog'});

        % Save
        my_filename = [filename '_' erase(num2str(training_sample_percent),".") ...
            '_' combine_rule{i} '_' select_weight_type];
        my_save_path = [save_path filesep my_filename];
        eval([my_filename ' = data_log;']);
        save(my_save_path, my_filename,'-v7.3');
        eval(['clear ' my_filename])
        clear data_log
    end
    
end

