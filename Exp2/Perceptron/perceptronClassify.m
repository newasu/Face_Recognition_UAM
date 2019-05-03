function [predictY, accuracy, mdl, scores, trainingTime, testTime] = perceptronClassify(...
    trainingDataX, trainingDataY, testDataX, testDataY, testFileNames, varargin)
%PERCEPTRONCLASSIFY Summary of this function goes here
%   Detailed explanation goes here

	seed = getAdditionalParam( 'seed', varargin, 1 );

    class_name = categorical(categories(trainingDataY));
    [new_trainingDataY, ~] = grp2idx(trainingDataY);
    
%     net = perceptron();
    net = patternnet([]); % [] for no hidden layer
%     net.trainParam.showWindow = false;
%     net.trainParam.epochs = 100;
%     SetRandomSeed(seed);
    
    tic
    mdl = train(net, trainingDataX', new_trainingDataY');
    trainingTime = toc;
    
    tic
    [predictY] = mdl(testDataX');
    testTime = toc;
    
%     Convert predictY back to actual label
    predictY = class_name(round(predictY));
    
    accuracy = (sum(predictY==testDataY)/numel(testDataY)) * 100;
    scores = table(testFileNames, testDataY, predictY, 'VariableNames', {'filenames' 'labels', 'predict_labels'});
    
end

