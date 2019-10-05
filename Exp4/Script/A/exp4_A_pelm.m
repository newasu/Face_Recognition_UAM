% exp3 A classify Is that same person by euclidean PELM2

clear all

% Settings
experiment_name = 'Exp4';
sub_experiment_name = 'A';
method_name = 'pelm';

random_seed_experimant = [1 2 3 4 5 6 7 8 9 10]; % number of experiment tested
number_sub_dataset = 10;
numb_cv = 5; % cv for optimal parameters
selected_pose_numb = 3; % number of image used each user
number_comparison = 1; % number of pair comparison for each same class

% PELM's parameters
hiddenNodes = 10:10:100;
regularizationC = power(10,-6:1:6);
select_weight_type = 'random_select'; % random_select random_generate
distFunction = 'euclidean'; % euclidean cosine
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
        diveface_feature(temp,:) = [];
    end
end
clear contain_image id i temp

% Generate training/test set
[subdataset, subdataset_label] = SplitSubdataset(diveface_label.id, ...
    {diveface_label.gender, diveface_label.ethnicity}, ...
    'number_sub_dataset', 10);

