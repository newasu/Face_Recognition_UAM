function [arranged_perf_mat] = exp3_arrange_method_class(method_name, perf_mat)
%EXP3_ARRANGE_METHOD_CLASS Summary of this function goes here
%   Detailed explanation goes here

    eval_name = perf_mat{1, 1}.Properties.VariableNames;
    class_name = perf_mat{1, 1}.Properties.RowNames;
    
    for k = 1 : numel(eval_name)
        temp = [];
        for i = 1 : numel(class_name)
            for j = 1 : numel(method_name)
                temp_method_name = method_name{j};
                temp_score = perf_mat{j};     
                temp(i,j) = temp_score.(eval_name{k})(class_name{i});
            end
        end
        
        arranged_perf_mat.(eval_name{k}) = array2table(temp, ...
            'VariableNames', method_name, 'RowNames', class_name);
    end

end

