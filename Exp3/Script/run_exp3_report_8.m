
pos_class = 'same';

% load data
result_path = [pwd '_data_store/Result/Exp3'];
my_file_path = dir([result_path '/**/*.mat']);
my_file_path = my_file_path(contains({my_file_path.name}, '_01'));

method_name = struct2cell(my_file_path);
method_name = method_name(1,:)';
method_name = cellfun(@(x) x(8:strfind(x, '.')-1), ...
        method_name, 'UniformOutput', false);
% method_name = table(method_name, 'variablenames', {'method_name'});

[~, diveface_label] = LoadDiveFaceFull();

% Evaluate performance matrix
biometric_perf_mat = [];
for i = 1 : numel(my_file_path)
    temp = load([my_file_path(i).folder '/' my_file_path(i).name]);
    disp(['loaded :' num2str(i) '/' num2str(numel(my_file_path))]);
    
    fname = fieldnames(temp);
    temp = temp.(fname{1}).testResult.label_mat{1};
    temp_filename = temp.filenames;
    temp_true_label = temp.labels;
    temp_predict_label = temp.predict_labels;
    temp_label = unique(temp_true_label);
    temp_pos_label_idx = find(strcmp(cellstr(temp_label), pos_class));
    if size(temp.predict_score,2) > 1
        temp_pos_score = temp.predict_score(:, temp_pos_label_idx);
        pos_score_order = 'ascend';
        temp_pos_class_idx = find(strcmp(cellstr(temp_label), pos_class));
    else
        temp_pos_score = temp.predict_score;
        temp_pos_score(find(temp_pos_score == inf)) = 0;
        pos_score_order = 'descend';
        temp_pos_class_idx = find(~strcmp(cellstr(temp_label), pos_class));
    end

    temp_biometric_perf_mat = exp3_report_6(...
        diveface_label, temp_filename, temp_true_label, temp_predict_label, temp_pos_score, ...
        'positive_class_score_order', pos_score_order);
    
    biometric_perf_mat = [biometric_perf_mat; {temp_biometric_perf_mat} ];

end

[arranged_biometric_perf_mat] = exp3_arrange_method_class(method_name, biometric_perf_mat);

clear result_path my_file_path i fname pos_class method_name diveface_label

clear eer far frr threshold_vector fmr0d1 fmr0d01
clear temp temp_true_label temp_label temp_pos_class_idx temp_pos_score temp_predict_label
clear temp_pos_label_idx temp_filename pos_score_order
clear impostor_idx genuine_idx impostor_data genuine_data
clear biometric_perf_threshold temp_biometric_perf_mat

