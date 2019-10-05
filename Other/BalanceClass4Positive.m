function [new_data_X, new_data_Y] = BalanceClass4Positive(data_X, data_Y, data_info, varargin)
%BALANCECLASS4POSITIVE Summary of this function goes here
%   Detailed explanation goes here

    seed = getAdditionalParam( 'seed', varargin, 1 );
    pos_class_name = getAdditionalParam( 'pos_class_name', varargin, 'same' );

    unique_class = unique(data_Y);
    number_member = hist(data_Y, unique_class);
    [~, min_member_idx] = min(number_member);
    min_member_class = unique_class(min_member_idx);
    pos_idx = data_Y == pos_class_name;
    neg_idx = ~pos_idx;
    
    data_X_id = cell2mat(arrayfun(@(x) data_info.id(find(data_info.data_id==x)), ...
        data_X(:,1), 'UniformOutput', false));
    data_X_gender = arrayfun(@(x) data_info.gender(find(data_info.data_id==x)), ...
        data_X(:,2), 'UniformOutput', false);
    data_X_ethnicity = arrayfun(@(x) data_info.ethnicity(find(data_info.data_id==x)), ...
        data_X(:,2), 'UniformOutput', false);
    mate_class = cellfun(@(x,y) strcat(cellstr(x), '_', cellstr(y)), ...
        data_X_gender, data_X_ethnicity, 'UniformOutput', false);
    mate_class = cellfun(@(x) x{:}, mate_class, 'UniformOutput', false);
    mate_class = categorical(mate_class);
    unique_mate_class = categorical(unique(mate_class));
    data_X_id = categorical(data_X_id);
    data_label = table(data_X, data_X_id, data_Y, mate_class);
    unique_data_id = unique(data_X_id);
    
    temp = data_label(pos_idx,:);
    pos_min = hist(temp.data_X_id);
    pos_min = min(pos_min);
    
    new_data_X = [];
    new_data_Y = [];
    
    for i = 1 : size(unique_data_id,1)
        temp = data_label.data_X_id == unique_data_id(i);
        temp = data_label(temp,:);
        
        temp_idx = temp.data_Y == pos_class_name;
        temp_pos = temp(temp_idx,:);
        temp_neg = temp(~temp_idx,:);
        
        SetRandomSeed(seed);
        collect_idx = randperm(size(temp_pos,1));
        seed = seed + 1;
        collect_idx = collect_idx(1:pos_min);
        
        temp_pos = temp_pos(collect_idx,:);
        
        % Fix this, random pick for each class in unique_mate_class
        SetRandomSeed(seed);
        collect_idx = randperm(size(temp_neg,1));
        seed = seed + 1;
        collect_idx = collect_idx(1:size(unique_mate_class,1));
    end
    
    

end

