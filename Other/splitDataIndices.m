function idxTable = splitDataIndices(myLabels, varargin)
%SPLITDATAINDICES Summary of this function goes here
%   Detailed explanation goes here
% idxTable = splitDataIndices(filteredData.label, 0.7, 1);

    proportion = getAdditionalParam( 'proportion', varargin, 0.7 );
    randomSeed = getAdditionalParam( 'randomSeed', varargin, 1 );

    if iscategorical(myLabels)
        myLabels = cellstr(myLabels);
    end
    
    [idxLabels, labelNames] = grp2idx(myLabels);
    uniqueIdxLabels = unique(idxLabels);
    countLabels = hist(idxLabels, uniqueIdxLabels);
    
    labelIdx = arrayfun(@(x) find(idxLabels == x), uniqueIdxLabels, 'UniformOutput', false);
    randNumb = arrayfun(@(x) myRandom(x,randomSeed), countLabels, 'UniformOutput', false);
    [trainingRandNumb, testRandNumb] = cellfun(@(x) divideProportion(x, proportion), randNumb, 'UniformOutput', false);
    trainingIdx = cellfun(@(x,y) x(y)', labelIdx, trainingRandNumb', 'UniformOutput', false);
    testIdx = cellfun(@(x,y) x(y)', labelIdx, testRandNumb', 'UniformOutput', false);
    
    idxTable = table(labelNames,trainingIdx,testIdx);
end

function rn = myRandom(elementNumb, seed)
    rng(seed);
    rn = randperm(elementNumb);
end

function [firstGroupIdx, secondGroupIdx] = divideProportion(data, prop)
    dataSize = numel(data);
    firstGroupSize = round(dataSize * prop);
    firstGroupIdx = data(1:firstGroupSize);
    secondGroupIdx = data((firstGroupSize + 1) : dataSize);
end