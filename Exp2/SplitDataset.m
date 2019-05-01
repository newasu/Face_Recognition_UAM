function [training_id_index, test_id_index, data, user_index] =...
    SplitDataset(data_label, varargin)
%SplitDataset Summary of this function goes here
%   Detailed explanation goes here
% [training_index, test_index, data, user_index] = SplitDataset(raw_label);

%     group_by = getAdditionalParam('group_by', varargin, 'gender'); % gender ethnicity
    training_sample_percent = getAdditionalParam('training_sample_percent', varargin, 0.9);
    random_seed = getAdditionalParam('random_seed', varargin, 1); % set to 999 for freely random
        
    gender_cat = categorical(categories(data_label.gender));
    ethnicity_cat = categorical(categories(data_label.ethnicity));

% Reconstruct data structure
    my_data_label = sortrows(data_label, {'id', 'pose'});
    id = unique(my_data_label.id);
    gender = [];
    ethnicity = [];
    user_index = [];
    for i = 1 : numel(id)
        temp = find(data_label.id == id(i));
        user_index{i} = temp;
        gender(i) = data_label.gender(temp(1));
        ethnicity(i) = data_label.ethnicity(temp(1));
    end   

% Label
    mylabel = strcat(string(gender), string(ethnicity));
    [mylabel, ~] = grp2idx(mylabel);
    unique_mylabel = unique(mylabel);
    count_label = hist(mylabel, unique_mylabel);

% Random
    label_index = arrayfun(@(x) find(mylabel == x), unique_mylabel, 'UniformOutput', false);
    random_index_data = arrayfun(@(x) myRandom(x,random_seed), count_label, 'UniformOutput', false);
    [trainingRandNumb, testRandNumb] = cellfun(@(x)...
        divideProportion(x, training_sample_percent), random_index_data, 'UniformOutput', false);
    training_index = cellfun(@(x,y) x(y)', label_index, trainingRandNumb', 'UniformOutput', false);
    test_index = cellfun(@(x,y) x(y)', label_index, testRandNumb', 'UniformOutput', false);
    
% Bind random index to id
    training_id_index = cellfun(@(x) id(x), training_index, 'UniformOutput', false);
    test_id_index = cellfun(@(x) id(x), test_index, 'UniformOutput', false);
    
% Bind into table
    data = table(gender_cat(gender), ethnicity_cat(ethnicity), id,...
        'VariableNames', {'gender', 'ethnicity', 'id'});
    
end

function rn = myRandom(elementNumb, seed)
    SetRandomSeed(seed);
    rn = randperm(elementNumb);
end

function [firstGroupIdx, secondGroupIdx] = divideProportion(data, prop)
    dataSize = numel(data);
    firstGroupSize = round(dataSize * prop);
    firstGroupIdx = data(1:firstGroupSize);
    secondGroupIdx = data((firstGroupSize + 1) : dataSize);
end
