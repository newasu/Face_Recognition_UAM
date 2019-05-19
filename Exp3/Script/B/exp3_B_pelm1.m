% exp3 B classify Is that same person by linear PELM1

clear all

% Settings
experiment_name = 'Exp3';
sub_experiment_name = 'B';
method_name = 'pelm1';

numb_run = 1;
numb_cv = 5;
training_sample_percent = 0.001; % percentages of training sample
selected_pose_numb = 3; % number of image used each user
number_comparison = 1; % number of pair comparison for each same class
do_balance_class = 1;

% PELM's parameters
hiddenNodes = 10:10:100;
regularizationC = power(10,-6:1:6);
select_weight_type = 'random_select'; % random_select random_generate
distFunction = 'cosine'; % euclidean cosine
combine_rule = {'distance', 'mean', 'multiply', 'sum'}; % distance mean multiply sum

% Save path
default_data_store_path = pwd;
idcs = strfind(pwd,filesep);
default_data_store_path = [default_data_store_path(1:idcs(end)-1) ...
    filesep 'Face_Recognition_UAM_data_store'];
saveFolderPath = {'Result', experiment_name, [experiment_name '_' sub_experiment_name]};
filename = [saveFolderPath{end} '_' method_name];
save_path = MakeChainFolder(saveFolderPath, 'target_path', default_data_store_path);

clear idcs default_data_store_path saveFolderPath

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

for combine_rule_round = 1 : numel(combine_rule)
    for random_seed = 1 : numb_run
        % Split dataset
        [training_id_index, test_id_index, data, user_index] = SplitDataset(diveface_label,...
            'training_sample_percent', training_sample_percent, 'random_seed', random_seed);
        
%         Subsampling test_id_index size to be equal to training_id_index size
        for i = 1 : size(test_id_index,1)
            for j = 1 : size(test_id_index,2)
                temp_train = training_id_index{i,j};
                temp_train = numel(temp_train{1});
                temp_test = test_id_index{i,j};
                temp_test = temp_test{1};
                temp_test = temp_test(randperm(numel(temp_test), temp_train));
                
                test_id_index{i,j} = {temp_test};
            end
        end
        
        clear temp_train temp_test i j

        [training_pair_list, training_label_pair_list] = PairDataEachClass(...
            training_id_index, diveface_label,...
            'random_seed', random_seed, 'number_comparison', number_comparison, ...
            'selected_pose_numb', selected_pose_numb);
        [test_pair_list, test_label_pair_list] = PairDataEachClass(...
            test_id_index, diveface_label,...
            'random_seed', random_seed, 'number_comparison', number_comparison, ...
            'selected_pose_numb', selected_pose_numb);

        % Subsampling to balance dataset
        if do_balance_class
            [training_pair_list, training_label_pair_list] = BalanceClasses(...
                training_pair_list, training_label_pair_list, 'random_seed', random_seed);
            [test_pair_list, test_label_pair_list] = BalanceClasses(...
                test_pair_list, test_label_pair_list, 'random_seed', random_seed);
        end

        % Training data
        trainingDataX_1 = diveface_feature(training_pair_list(:,1),:);
        trainingDataX_2 = diveface_feature(training_pair_list(:,2),:);
        training_data_id = training_pair_list;
        trainingDataY = training_label_pair_list;
        
        % Test data
        testDataX_1 = diveface_feature(test_pair_list(:,1),:);
        testDataX_2 = diveface_feature(test_pair_list(:,2),:);
        test_data_id = test_pair_list;
        testDataY = test_label_pair_list;

        % Genarate k-fold indices
        [ kFoldIdx, ~ ] = GetKFoldIndices( numb_cv, trainingDataY, random_seed );

        % Find optimal parameter
        [ foldLog, avgFoldLog ] = pelm1CV(kFoldIdx, diveface_feature, ...
            trainingDataX_1, trainingDataX_2, trainingDataY, training_data_id, ...
            'seed', random_seed, 'hiddenNodes', hiddenNodes, 'regularizationC', ...
            regularizationC, 'distFunction', distFunction, 'combine_rule', ...
            combine_rule{combine_rule_round}, 'select_weight_type', select_weight_type);
        
        % Test model
        [trainingResult, testResult, testCorrectIdx] = TestPELM1Params(diveface_feature, ...
            trainingDataX_1, trainingDataX_2, trainingDataY, training_data_id, ...
            testDataX_1, testDataX_2, testDataY, test_data_id, ...
            'hiddenNodes', table2array(avgFoldLog(1,1)), 'regularizationC', ...
            table2array(avgFoldLog(1,2)), 'seed', random_seed, ...
            'distFunction', distFunction, 'select_weight_type', select_weight_type, ...
            'combine_rule', combine_rule{combine_rule_round});

        data_log(random_seed,:) = {foldLog trainingResult testResult};
        
        clear training_id_index test_id_index data user_index
        clear training_pair_list training_label_pair_list
        clear test_pair_list test_label_pair_list
        clear trainingDataX_1 trainingDataX_2 trainingDataY training_data_id
        clear testDataX_1 testDataX_2 testDataY test_data_id
        clear kFoldIdx foldLog avgFoldLog trainingResult testResult 
    end
    
    % Save
    data_log = cell2table(data_log, 'variablenames', ...
        {'foldLog' 'trainingResult' 'testResult'});
    my_filename = [filename '_' erase(num2str(training_sample_percent),".") ...
        '_' combine_rule{combine_rule_round} '_' select_weight_type];
    my_save_path = [save_path filesep my_filename];
    eval([my_filename ' = data_log;']);
    save(my_save_path, my_filename,'-v7.3');
    eval(['clear ' my_filename])
    clear data_log my_filename my_save_path
end
