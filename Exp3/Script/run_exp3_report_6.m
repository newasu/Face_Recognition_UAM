
% load data
result_path = '/Users/Wasu/Google Drive/MacBook/PhD''s Degree/New/SourceCode/Face_Recognition_UAM_data_store/Result/Exp3';
my_file_path = dir([result_path '/**/*.mat']);
my_file_path = my_file_path(contains({my_file_path.name}, '_01'));

[~, diveface_label] = LoadDiveFaceFull();

accuracy_mat = [];
auc_mat = [];
eer_mat = [];
for i = 1 : numel(my_file_path)
    load([my_file_path(i).folder '/' my_file_path(i).name])
    disp(['loaded :' num2str(i) '/' num2str(numel(my_file_path))]);
    
    data = load([my_file_path(i).folder '/' my_file_path(i).name]);
    fname = fieldnames(data);
    data = data.(fname{1}).testResult.label_mat{1};
    my_filename = data.filenames;
    my_true_label = data.labels;
    my_predict_label = data.predict_labels;
    
    % Convert score if Euclidean distance
    if contains(my_file_path(i).name, 'C_dist')
        my_predict_score = 1./data.predict_score;
    else
        my_predict_score = data.predict_score(:,2);
    end
    
    [data_class, data_accuracy, data_auc, data_eer] = exp3_report_6(...
        diveface_label, my_filename, my_true_label, my_predict_label, my_predict_score);
    
    accuracy_mat = [accuracy_mat data_accuracy];
    auc_mat = [auc_mat data_auc];
    eer_mat = [eer_mat data_eer];
    
    disp(['calculated :' num2str(i) '/' num2str(numel(my_file_path))]);
end

method_name = {my_file_path.name}';

clear result_path my_file_path i fname
clear diveface_label data fname 
clear my_filename my_true_label my_predict_label my_predict_score
clear data_accuracy data_auc data_eer

