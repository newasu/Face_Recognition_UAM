function [predictY, accuracy, mdl, scores, trainingTime, testTime] = mlpClassify(...
    trainingDataX, trainingDataY, testDataX, testDataY, testFileNames, varargin)
%MLPCLASSIFY Summary of this function goes here
%   Detailed explanation goes here

    transferFunction = getAdditionalParam( 'transferFunction', varargin, 'tansig' );  % softmax tansig logsig
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    seed = getAdditionalParam( 'seed', varargin, 1 );
    
    hiddenNodes_percentages = round((hiddenNodes/100) * numel(trainingDataY));

    net = feedforwardnet(hiddenNodes_percentages, 'trainrp');
    net.layers{numel(net.layers)}.transferFcn = transferFunction;
%     net.trainParam.epochs = 1000;
    net.trainParam.showWindow = false;
%     SetRandomSeed(seed)
    
    tic
    mdl = train(net, trainingDataX', trainingDataY');
    trainingTime = toc;
    
    tic
    predictY = mdl(testDataX')';
    testTime = toc;
    
    accuracy = (sum(round(predictY)==testDataY)/numel(testDataY)) * 100;
    scores = table(testFileNames, testDataY, predictY, 'VariableNames', {'filenames' 'labels', 'predict_labels'});
end

