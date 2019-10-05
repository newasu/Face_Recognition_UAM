function [subdataset, subdataset_label] = SplitSubdataset(sample_id, data_label, varargin)
%SPLITSUBDATASET Summary of this function goes here
%   Detailed explanation goes here

    number_sub_dataset = getAdditionalParam('number_sub_dataset', varargin, 10);
    random_seed = getAdditionalParam('random_seed', varargin, 1); % set to 999 for freely random
    
    % Prepare data
    unique_sample_id = unique(sample_id);
    frequency_sample_id = min(hist(sample_id, unique_sample_id));
    my_sample_id = reshape(sample_id, frequency_sample_id, numel(unique_sample_id));
    my_sample_id = my_sample_id(1,:);
    for i = 1 : size(data_label,2)
        my_cat{i} = reshape(data_label{i}, frequency_sample_id, numel(unique_sample_id));
        my_cat{i} = my_cat{i}(1,:);
        my_unique_cat{i} = categorical(categories(my_cat{i}));
    end

    % Split data into its class
    [my_splited_classes, my_splited_classes_label] = SplitConcatClasses(my_sample_id, my_cat);
    
    % Random data
    my_random_seed = random_seed;
    for i = 1 : numel(my_splited_classes_label)
        ttemp = sum(~isnan(my_splited_classes(:,i)));
        temp = myRandom(ttemp, my_random_seed);
        my_splited_classes(1:ttemp,i) = my_splited_classes(temp,i);
        my_random_seed = my_random_seed + 1;
    end
    
    % Bind into subdatasets
    subdataset_split_size = size(my_splited_classes,1)/number_sub_dataset;
    begin_idx = 1;
    for i = 1 : number_sub_dataset
        subdataset{i} = my_splited_classes(begin_idx:(begin_idx+subdataset_split_size-1),:);
        begin_idx = begin_idx + subdataset_split_size;
    end
    
    subdataset_label = my_splited_classes_label;

end

function rn = myRandom(elementNumb, seed)
    SetRandomSeed(seed);
    rn = randperm(elementNumb);
end

function [splited_classes, unique_concat_label] = SplitConcatClasses(my_sample_id, my_cat)
    concat_label = string(my_cat{1});
    for ii = 2 : numel(my_cat)
        concat_label = strcat(concat_label, '_', string(my_cat{ii}));
    end
    
    [my_concat_label, unique_concat_label] = grp2idx(concat_label);
    unique_my_concat_label = unique(my_concat_label);
    count_label = hist(my_concat_label, unique_my_concat_label);
    
    splited_classes = nan(max(count_label), numel(unique_my_concat_label));
    for ii = 1 : numel(unique_my_concat_label)
        ttemp = my_concat_label == unique_my_concat_label(ii);
        splited_classes(1:sum(ttemp),ii) = my_sample_id(ttemp);
    end
end


