
positive_class = 'same';

%% load data

result_path = '/Users/Wasu/Google Drive/MacBook/PhD''s Degree/New/SourceCode/Face_Recognition_UAM_data_store/Result/Exp3';
my_file_path = dir([result_path '/**/*.mat']);

% for i = 1 : numel(my_file_path)
%     load([my_file_path(i).folder '/' my_file_path(i).name])
%     disp(['loaded :' num2str(i) '/' num2str(numel(my_file_path))]);
% end

clear result_path my_file_path i

my_data{1,1} = Exp3_A_pelm2_0001_distance_random_select;
my_data{2,1} = Exp3_A_pelm2_0005_distance_random_select;
my_data{3,1} = Exp3_A_pelm2_001_distance_random_select;
my_data{4,1} = Exp3_A_pelm2_005_distance_random_select;
my_data{5,1} = Exp3_A_pelm2_01_distance_random_select;

my_data{1,2} = Exp3_A_pelm2_0001_mean_random_select;
my_data{2,2} = Exp3_A_pelm2_0005_mean_random_select;
my_data{3,2} = Exp3_A_pelm2_001_mean_random_select;
my_data{4,2} = Exp3_A_pelm2_005_mean_random_select;
my_data{5,2} = Exp3_A_pelm2_01_mean_random_select;

my_data{1,3} = Exp3_A_pelm2_0001_multiply_random_select;
my_data{2,3} = Exp3_A_pelm2_0005_multiply_random_select;
my_data{3,3} = Exp3_A_pelm2_001_multiply_random_select;
my_data{4,3} = Exp3_A_pelm2_005_multiply_random_select;
my_data{5,3} = Exp3_A_pelm2_01_multiply_random_select;

my_data{1,4} = Exp3_A_pelm2_0001_sum_random_select;
my_data{2,4} = Exp3_A_pelm2_0005_sum_random_select;
my_data{3,4} = Exp3_A_pelm2_001_sum_random_select;
my_data{4,4} = Exp3_A_pelm2_005_sum_random_select;
my_data{5,4} = Exp3_A_pelm2_01_sum_random_select;

my_data{1,5} = Exp3_D_celm_0001_random_select;
my_data{2,5} = Exp3_D_celm_0005_random_select;
my_data{3,5} = Exp3_D_celm_001_random_select;
my_data{4,5} = Exp3_D_celm_005_random_select;
my_data{5,5} = Exp3_D_celm_01_random_select;

my_data{1,6} = Exp3_C_dist_0001;
my_data{2,6} = Exp3_C_dist_0005;
my_data{3,6} = Exp3_C_dist_001;
my_data{4,6} = Exp3_C_dist_005;
my_data{5,6} = Exp3_C_dist_01;

%% ELM

accuracy = [];
auc = [];
eer = [];
for i = 1 : size(my_data, 1)
    for j = 1 : size(my_data, 2)
        
        temp_data = my_data{i,j}.testResult(1,:).label_mat{1};
        data_true_label = temp_data.labels;
        data_predict_label = temp_data.predict_labels;
        my_label = unique(data_true_label);
        data_score = temp_data.predict_score;
        
        if size(data_score, 2) > 1
            data_score = data_score(:, find(my_label == positive_class));
        else
            data_score = 1./data_score;
        end
        
        [temp_accuracy, temp_auc, temp_eer] = exp3_report_auc_eer(data_true_label, ...
            data_predict_label, data_score, positive_class);
        
        accuracy(i,j) = temp_accuracy;
        auc(i,j) = temp_auc;
        eer(i,j) = temp_eer;
    end
end

clear my_data i j temp_data 
clear data_true_label data_predict_label my_label data_score
clear temp_accuracy temp_auc temp_eer

%% Euclidean

% my_data{1} = Exp3_C_dist_0001;
% my_data{2} = Exp3_C_dist_0005;
% my_data{3} = Exp3_C_dist_001;
% my_data{4} = Exp3_C_dist_005;
% my_data{5} = Exp3_C_dist_01;
% 
% for i = 1 : numel(my_data)
%         
%     temp_data = my_data{i}.testResult(1,:).label_mat{1};
%     data_true_label = temp_data.labels;
%     data_predict_label = temp_data.predict_labels;
%     my_label = unique(data_true_label);
%     data_score = temp_data.predict_score;
%     
%     temp_accuracy(i) = sum(data_true_label==data_predict_label) / numel(data_true_label);
%     
%     impostor_vector = data_score(data_true_label=='different');
%     genuine_vector = data_score(data_true_label=='same');
%     [equal_error_rate] = eer_plot(genuine_vector, impostor_vector,0);
% 
%     temp_auc(i) = nan;
%     temp_eer(i) = equal_error_rate;
% 
% end
% 
% accuracy(:,end+1) = temp_accuracy;
% auc(:,end+1) = temp_auc;
% eer(:,end+1) = temp_eer;
% 
% clear my_data i temp_data
% clear impostor_vector genuine_vector equal_error_rate
% clear data_true_label data_predict_label my_label data_score
% clear temp_accuracy temp_auc temp_eer

clear positive_class
