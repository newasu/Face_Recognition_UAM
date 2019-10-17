function [kFoldIdx_id, kFoldIdx_data_id] = GetKFoldIdxSepID(k, sample_id, randSeed)
%GETKFOLDIDXSEPID Summary of this function goes here
%   Detailed explanation goes here

    unique_sample_id = unique(sample_id);
    rn = myRandom(numel(unique_sample_id), randSeed);
    unique_sample_id = unique_sample_id(rn);
    
    % Mod to balance each fold
    balance_set = mod(numel(unique_sample_id),k);
    unique_sample_id = unique_sample_id(1:end-balance_set);
    
    kFoldIdx_id = reshape(unique_sample_id, [numel(unique_sample_id)/k k])';
    
    kFoldIdx_data_id = [];
    for ii = 1 : numel(unique_sample_id)
        kFoldIdx_data_id = [kFoldIdx_data_id; find(sample_id == unique_sample_id(ii))];
    end
    kFoldIdx_data_id = reshape(kFoldIdx_data_id, [numel(kFoldIdx_data_id)/k k])';
end

function rn = myRandom(elementNumb, seed)
    SetRandomSeed(seed);
    rn = randperm(elementNumb);
end