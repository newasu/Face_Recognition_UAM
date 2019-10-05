function [arranged_perf_mat] = exp3_arrange_method_percentage(perf_mat)
%EXP3_ARRANGE_METHOD_PERCENTAGE Summary of this function goes here
%   Detailed explanation goes here

    eval_name = perf_mat.Properties.VariableNames;
    method_name = perf_mat.(eval_name{1});
    eval_name(1) = [];
    method_name = strcat(method_name,'_');
    sep_idx = strfind(method_name, '_');
    
    my_method_prefix = cellfun(@(x,y) x(1:y(1)), ...
        method_name, sep_idx, 'UniformOutput', false);
    my_method_percent = cellfun(@(x,y) x(y(1)+1:y(2)), ...
        method_name, sep_idx, 'UniformOutput', false);
    my_method_suffix = cellfun(@(x,y) x(y(2)+1:end), ...
        method_name, sep_idx, 'UniformOutput', false);
    my_method_name = strcat(my_method_prefix, my_method_suffix);
    my_method_name = cellfun(@(x) x(1:end-1), ...
        my_method_name, 'UniformOutput', false);
    my_method_name = unique(my_method_name);
    my_percent = unique(my_method_percent);
    my_percent = cellfun(@(x) x(1:end-1), ...
        my_percent, 'UniformOutput', false);
    clear sep_idx temp
    
    for k = 1 : numel(eval_name)
        temp = [];
        for i = 1 : numel(my_percent)
            for j = 1 : numel(my_method_name)
                temp_method_name = my_method_name{j};
                temp_method_name_idx = strfind(temp_method_name, '_');
                if ~isempty(temp_method_name_idx)
                    temp_method_name = [temp_method_name(1:temp_method_name_idx(1)) ...
                        my_percent{i} '_' temp_method_name(temp_method_name_idx(1)+1:end)];
                else
                    temp_method_name = [temp_method_name '_' my_percent{i}];
                end
                
                temp_idx = find(contains(method_name, temp_method_name));
                
                temp(i,j) = perf_mat.(eval_name{k})(temp_idx);
            end
        end
        
        arranged_perf_mat.(eval_name{k}) = array2table(temp, ...
            'VariableNames', my_method_name, 'RowNames', my_percent);
    end
    
end

