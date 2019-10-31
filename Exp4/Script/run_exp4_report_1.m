
% Exp4_A_pelm_euclidean_distance_randomselect Exp4_B_dist_euclidean

data_name = 'Exp4_B_dist_euclidean';
data_path = [pwd '_data_store/Result/Exp4'];

% Exact
[data_result, data_result_all] = exp4_exact_result(data_name, data_path);
[~, diveface_label] = LoadDiveFaceFull();

% Prepare data
pos_class = 'same';
temp_filename = data_result_all.filenames;
temp_true_label = data_result_all.labels;
temp_predict_label = data_result_all.predict_labels;
temp_label = unique(temp_true_label);
temp_pos_label_idx = find(strcmp(cellstr(temp_label), pos_class));
if size(data_result_all.predict_score,2) > 1
    temp_pos_score = data_result_all.predict_score(:, temp_pos_label_idx);
    pos_score_order = 'ascend';
    temp_pos_class_idx = find(strcmp(cellstr(temp_label), pos_class));
else
    temp_pos_score = data_result_all.predict_score;
    temp_pos_score((temp_pos_score == inf)) = 0;
    pos_score_order = 'descend';
    temp_pos_class_idx = find(~strcmp(cellstr(temp_label), pos_class));
end

% Evaluate
biometric_perf_mat = exp3_report_6(...
    diveface_label, temp_filename, temp_true_label, temp_predict_label, temp_pos_score, ...
    'positive_class_score_order', pos_score_order);

% Save
default_data_store_path = pwd;
idcs = strfind(pwd,filesep);
default_data_store_path = [default_data_store_path(1:idcs(end)-1) ...
    filesep 'Face_Recognition_UAM_data_store'];
saveFolderPath = {'Result', 'Exp4', 'Exp4_average'};
save_path = MakeChainFolder(saveFolderPath, 'target_path', default_data_store_path);
my_data_name = [data_name '_average_class'];
my_save_path = [save_path filesep my_data_name];
eval([my_data_name ' = biometric_perf_mat;']);
save(my_save_path, my_data_name,'-v7.3');
eval(['clear ' my_data_name])
clear default_data_store_path idcs saveFolderPath save_path my_data_name my_save_path

clear pos_class temp_filename temp_true_label temp_predict_label temp_label temp_pos_label_idx temp_pos_score pos_score_order temp_pos_class_idx