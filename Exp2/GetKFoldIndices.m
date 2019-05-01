function [ kFoldIdx, idxTable ] = GetKFoldIndices( k, label, randSeed )
%GETKFOLDINDEX Summary of this function goes here
%   Detailed explanation goes here
    
    % Split data each class
    data = splitDataIndices(label, 'proportion', 1, 'randomSeed', randSeed);
    dataName = data.labelNames;
    data = data.trainingIdx;
    data = cellfun(@(x) x(:,1:(end-mod(end,k))), data, 'UniformOutput', false);

    % assign folds
    data = cellfun(@(x) num2cell(reshape(x, numel(x)/k, k), [1 k]), data, 'UniformOutput', false);
    data = reshape([data{:}],[k numel(dataName)])';
    data = cellfun(@(x) x', data, 'UniformOutput', false);
    idxTable = table(dataName, data, 'VariableNames', {'Labels', 'Folds'});
    
    for i = 1 : k
        kFoldIdx(i,:) = [idxTable.Folds{:,i}];
    end
end

