% exp4 A classify Is that same person by euclidean PELM2

clear all

% Settings
experiment_name = 'Exp4';
sub_experiment_name = 'A';
method_name = 'pelm';

do_experiment = [1 2 3 4 5 6 7 8 9 10]; % run experiment number
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

% Remove user containing less than contain_image
disp('Removing error users..');
contain_image = 3;
id = unique(diveface_label.id);
for i = 1 : numel(id)
    temp = find(diveface_label.id == id(i));
    if numel(temp) < contain_image
        diveface_label(temp,:) = [];
        diveface_feature(temp,:) = [];
    end
end
clear contain_image id i temp
disp('Removed error users');

% Generate training/test set
[subdataset, subdataset_label] = SplitSubdataset(diveface_label.id, ...
    {diveface_label.gender, diveface_label.ethnicity}, ...
    'number_sub_dataset', number_subdataset);

% Test
subdataset = cellfun(@(x) x(1:50,:), subdataset, 'UniformOutput', false);

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
        
        % Load Paired sample list
        temp_paired_list_save_path = [paired_list_save_path filesep 'paired_list_' num2str(experiment_round)];
        if exist([temp_paired_list_save_path '.mat'], 'file')
            load(temp_paired_list_save_path);
        else
            [training_pair_list, training_pair_list_label] = PairSampleClassesEqually(...
                training_set_sample_idx, subdataset_label, diveface_label,...
                'random_seed', experiment_round);
            save(temp_paired_list_save_path, 'training_pair_list', 'training_pair_list_label','-v7.3');
        end
        
    end
end


