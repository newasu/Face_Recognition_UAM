function [my_result, my_result_all] = exp4_exact_result(data_name, data_path, varargin)
%exp4_exact_result Summary of this function goes here
%   Detailed explanation goes here

    data_type = getAdditionalParam( 'data_type', varargin, 'testResult' ); % trainingResult testResult

    % get file paths
    result_path = data_path;
    my_file_path = dir([result_path '/**/*.mat']);
    my_file_path = my_file_path(contains({my_file_path.name}, data_name));
    my_file_path(contains({my_file_path.name}, 'average')) = [];

    my_seed = [];
    my_label = [];
    my_result_all = [];
    
    % load data
    for ii = 1 : numel(my_file_path)
        disp(['Loading ' my_file_path(ii).name '..']);
        temp = load([my_file_path(ii).folder '/' my_file_path(ii).name]);
        fn = fieldnames(temp);
        temp = temp.(fn{1});
        temp = temp.(data_type){1}.label_mat;
        my_result_all = [my_result_all; temp{1}];
        my_label = [my_label; temp];
        
        temp = findstr(my_file_path(ii).name, '_');
        temp = temp(end);
        temp = [temp+1 findstr(my_file_path(ii).name, '.')-1];
        my_seed = [my_seed; str2num(my_file_path(ii).name(temp(1):temp(2)))];
    end
    
    my_result = table(my_seed, my_label, 'variablenames', {'seed', 'label'});
    my_result = sortrows(my_result,'seed','ascend');
end

