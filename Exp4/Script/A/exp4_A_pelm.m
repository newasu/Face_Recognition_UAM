% exp4 A classify Is that same person by euclidean PELM2

clear all

% Settings

experiment_mode = 'test';
experiment_name = 'Exp4';
sub_experiment_name = 'A';
method_name = 'pelm';
do_experiment = [1]; % run experiment number

number_subdataset = 10;
numb_cv = 5; % cv for optimal parameters

% PELM's parameters
hiddenNodes = 10:10:100;
regularizationC = power(10,-6:1:6);
select_weight_type = 'random_select'; % random_select random_generate
distFunction = 'euclidean'; % euclidean cosine
all_combine_rule = {'distance', 'mean', 'multiply', 'sum'}; % distance mean multiply sum

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
contain_image = 3;
id = unique(diveface_label.id);
for i = 1 : numel(id)
    temp = find(diveface_label.id == id(i));
    if numel(temp) < contain_image
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

% Test
if strcmp(experiment_mode, 'test')
    subdataset = cellfun(@(x) x(1:5,:), subdataset, 'UniformOutput', false);
    disp('RUNNING IN TEST MODE!!');
end

for experiment_round = do_experiment
    for combine_rule = 1 : numel(all_combine_rule)
        
        % Assign training/test set
        training_set_sample_idx = [];
        test_set_sample_idx = [];
        for i = 1 : number_subdataset
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
            load(temp_paired_list_save_path);
        else
            training_paired_label = PairSampleClassesEqually(...
                training_set_sample_idx, subdataset_label, diveface_label,...
                'random_seed', experiment_round);
            save(temp_paired_list_save_path, 'training_paired_label', '-v7.3');
        end
        
        % Load test paired sample list
        temp_paired_list_save_path = [paired_list_save_path filesep ...
            'test_paired_label_' num2str(experiment_round)];
        if exist([temp_paired_list_save_path '.mat'], 'file') && ~strcmp(experiment_mode, 'test')
            load(temp_paired_list_save_path);
        else
            test_paired_label = PairSampleClassesEqually(...
                test_set_sample_idx, subdataset_label, diveface_label,...
                'random_seed', experiment_round);
            save(temp_paired_list_save_path, 'test_paired_label', '-v7.3');
        end
        
        % Training data
        trainingDataX_1 = cell2mat(arrayfun(@(x) find(x==diveface_data_id), ...
            training_pair_list(:,1), 'UniformOutput', false));
        trainingDataX_1 = diveface_feature(trainingDataX_1,:);
        trainingDataX_2 = cell2mat(arrayfun(@(x) find(x==diveface_data_id), ...
            training_pair_list(:,2), 'UniformOutput', false));
        trainingDataX_2 = diveface_feature(trainingDataX_2,:);
        training_data_id = training_pair_list;
        trainingDataY = training_pair_list_label;
        
        % Test data
        testDataX_1 = cell2mat(arrayfun(@(x) find(x==diveface_data_id), ...
            test_pair_list(:,1), 'UniformOutput', false));
        testDataX_1 = diveface_feature(testDataX_1,:);
        testDataX_2 = cell2mat(arrayfun(@(x) find(x==diveface_data_id), ...
            test_pair_list(:,2), 'UniformOutput', false));
        testDataX_2 = diveface_feature(testDataX_2,:);
        test_data_id = test_pair_list;
        testDataY = test_pair_list_label;

        % Genarate k-fold indices
        [ kFoldIdx, ~ ] = GetKFoldIndices( numb_cv, trainingDataY, experiment_round );
        
    end
end


