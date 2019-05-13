function [selected_pose_index, unselected_pose_index] =...
    RandomPickUserPose(user_id_index, data_label, user_index, varargin)
%RandomPickUserPose Summary of this function goes here
%   Detailed explanation goes here

    selected_pose_numb = getAdditionalParam('selected_pose_numb', varargin, 1);
    random_seed = getAdditionalParam('random_seed', varargin, 1); % set to 0 for freely random
    
    my_data_label = data_label;
    user_pose_numb = cellfun(@(x) numel(x), user_index, 'UniformOutput', false);
    user_id = unique(my_data_label.id);
    id_index = find(strcmp(my_data_label.Properties.VariableNames, 'id'));
    my_data_label = [double(table2array(data_label(:,{'gender', 'ethnicity'}))) ...
        table2array(data_label(:,{'id', 'pose'}))];
    
    global my_random_seed
    my_random_seed = random_seed;
    
% Random pick user's pose
    random_index_data = cellfun(@(x) myRandom(x,random_seed), user_pose_numb, 'UniformOutput', false);
    random_index_data = cellfun(@(x,y) selectPoseNumber(x, selected_pose_numb, y),...
        random_index_data, user_pose_numb, 'UniformOutput', false);
    random_selected_pose_index = zeros(size(my_data_label,1),1);
    for i = 1 : numel(random_index_data)
        temp_index = find(my_data_label(:, id_index) == user_id(i));
        random_selected_pose_index(temp_index(random_index_data{i})) = 1;
    end
    clear temp_index

% Seperate user's pose
    temp_index = cellfun(@(x) find(ismember(my_data_label(:, id_index), x)),...
        table2cell(user_id_index), 'UniformOutput', false);
    [selected_pose_index, unselected_pose_index] = cellfun(@(x)...
        mysort(my_data_label(:, id_index), x, random_selected_pose_index),...
        temp_index, 'UniformOutput', false);
    
%     Bind into table
    gender_cat = user_id_index.Properties.RowNames;
    ethnicity_cat = user_id_index.Properties.VariableNames;
    temp_selected_pose_index = [];
    temp_unselected_pose_index = [];
    for i = 1 : size(selected_pose_index, 2)
        temp_selected_pose_index = [temp_selected_pose_index table(selected_pose_index(:, i),...
            'VariableNames', ethnicity_cat(i), 'RowNames', gender_cat)];
        temp_unselected_pose_index = [temp_unselected_pose_index table(unselected_pose_index(:, i),...
            'VariableNames', ethnicity_cat(i), 'RowNames', gender_cat)];
    end
    selected_pose_index = temp_selected_pose_index;
    unselected_pose_index = temp_unselected_pose_index;
    
    clear global my_random_seed
    
end

function rn = myRandom(elementNumb, seed)
    global my_random_seed
    SetRandomSeed(my_random_seed);
%     SetRandomSeed(seed);
    rn = randperm(elementNumb);
    my_random_seed = my_random_seed + 1;
end

function rn = selectPoseNumber(user_pose, selection_numb, pose_number)
    tt = min(selection_numb, pose_number);
    rn = user_pose(1:tt);
end

function [sp, usp] = mysort(dl, ti, si)
    temp_flag = zeros(numel(dl),1);
    temp_flag(ti) = 1;
    temp = [dl si temp_flag];
% Selected pose
    pose_idx = and(temp(:,2), temp(:,3));
    sp = find(pose_idx);
% Unselected pose
    pose_idx = and(~temp(:,2), temp(:,3));
    usp = find(pose_idx);
end

