function [predictY, score, mdl, label_mat, trainingTime, testTime] = mlpClassify(...
    trainingDataX, trainingDataY, testDataX, testDataY, testFileNames, varargin)
%MLPCLASSIFY Summary of this function goes here
%   Detailed explanation goes here

%     transferFunction = getAdditionalParam( 'transferFunction', varargin, 'tansig' );  % softmax tansig logsig
    hiddenNodes = getAdditionalParam( 'hiddenNodes', varargin, [100] );
    seed = getAdditionalParam( 'seed', varargin, 1 );
    
    hiddenNodes_percentages = round((hiddenNodes/100) * size(trainingDataX,2));
    class_name = categorical(categories(trainingDataY));
    [new_trainingDataY, ~] = grp2idx(trainingDataY);
%     
%     net = feedforwardnet(hiddenNodes_percentages, 'trainrp');
%     net.layers{numel(net.layers)}.transferFcn = transferFunction;
% %     net.trainParam.epochs = 1000;

    net = patternnet(hiddenNodes_percentages);
%     net = feedforwardnet(hiddenNodes_percentages);
    net.trainParam.showWindow = false;
    SetRandomSeed(seed)

    tic
    mdl = train(net, trainingDataX', ...
    	full(ind2vec(double(new_trainingDataY)')), 'useGPU','yes');
    trainingTime = toc;
    
    tic
    predictY = mdl(testDataX', 'useGPU','yes')';
    testTime = toc;
    
    predict_score = predictY;
    
%     Convert predictY back to actual label
    [~, predictY] = max(predictY, [], 2);
    predictY = class_name(predictY);
    
%     score = (sum(predictY==testDataY)/numel(testDataY)) * 100;
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    label_mat = table(testFileNames, testDataY, predictY, predict_score, ... 
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
end

