function [new_data_X, new_data_Y] = BalanceClasses(data_X, data_Y, varargin)
%BALANCECLASSES Summary of this function goes here
%   Detailed explanation goes here

    seed = getAdditionalParam( 'seed', varargin, 1 );

    unique_class = unique(data_Y);
    number_member = hist(data_Y, unique_class);
    min_member = min(number_member);
    remove_index = zeros(numel(data_Y),1);
    
    global my_random_seed
    my_random_seed = seed;
    
    for i = 1 : numel(number_member)
        temp_remove_index = find(data_Y==unique_class(i));
        rn = myRandom(number_member(i), min_member);
        temp_remove_index = temp_remove_index(rn);
        remove_index(temp_remove_index) = 1;
    end

    new_data_X = data_X(find(remove_index),:);
    new_data_Y = data_Y(find(remove_index),:);
    
    clear global my_random_seed
end

function rn = myRandom(elementNumb, limit)
    global my_random_seed
    SetRandomSeed(my_random_seed);
    rn = randperm(elementNumb, limit);
    my_random_seed = my_random_seed + 1;
end

