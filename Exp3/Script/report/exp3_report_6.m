function [biometric_perf_mat] = exp3_report_6(...
    dataset_table, label, true_label, predict_label, score, varargin)
%EXP3_REPORT_1 Summary of this function goes here
%   Detailed explanation goes here

    positive_class_score_order = getAdditionalParam( 'positive_class_score_order', varargin, 'ascend');

    gender_cat = categorical(categories(dataset_table.gender));
    ethnicity_cat = categorical(categories(dataset_table.ethnicity));
    
    data_idx = cell2mat(arrayfun(@(x) find(dataset_table.data_id == x), ...
        label, 'UniformOutput', false));
    data_gender = arrayfun(@(x) dataset_table.gender(x), ...
        data_idx, 'UniformOutput', false);
    data_ethnicity = arrayfun(@(x) dataset_table.ethnicity(x), ...
        data_idx, 'UniformOutput', false);
    data_label = cellfun(@(x,y) ...
        strcat(cellstr(x),'_',cellstr(y)), ...
        data_gender, data_ethnicity, 'UniformOutput', false);
    data_label = cellfun(@(x) x{:}, data_label, 'UniformOutput', false);
    
    biometric_perf_mat = [];
    temp_cat = [];
    my_class = [];
    for i = 1 : numel(ethnicity_cat)
        for j = 1 : numel(gender_cat)
            
            temp_cat = strcat(cellstr(gender_cat(j)),'_',cellstr(ethnicity_cat(i)));
            
            temp = strcmp(data_label, temp_cat);
            temp_idx = sum(temp,2) > 0;
            
            % Performance matrix
%             [temp_accuracy, temp_auc, temp_eer] = exp3_report_auc_eer(...
%                 true_label(temp_idx), predict_label(temp_idx), score(temp_idx), 'same');
            
            [~, temp_biometric_perf_mat] = exp3_report_biometric_perf(...
                true_label(temp_idx), predict_label(temp_idx), score(temp_idx), 'same', ...
                'positive_class_score_order', positive_class_score_order);
            
            my_class = [my_class; temp_cat];
            biometric_perf_mat = [biometric_perf_mat; temp_biometric_perf_mat];
        end
    end
    
    biometric_perf_mat.Properties.RowNames = my_class;
    
end

