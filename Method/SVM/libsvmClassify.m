function [predictY, score, mdl, label_mat, trainingTime, testTime] = libsvmClassify(...
    trainingDataX, trainingDataY, testDataX, testDataY, testFileNames, varargin)
%LIBSVMCLASSIFY Summary of this function goes here
%   Detailed explanation goes here

    s = checkParam(getAdditionalParam( 's', varargin, '' ), 's'); % type of SVM
    t = checkParam(getAdditionalParam( 't', varargin, '' ), 't'); % kernel function
    g = checkParam(getAdditionalParam( 'g', varargin, '' ), 'g'); % gamma
    c = checkParam(getAdditionalParam( 'c', varargin, '' ), 'c'); % parameter C
    n = checkParam(getAdditionalParam( 'n', varargin, '' ), 'n'); % parameter nu
    
    buildParam = [s t g c n '-q'];
    
    class_name = categorical(categories(trainingDataY));
    [new_trainingDataY, ~] = grp2idx(trainingDataY);

    if contains(t,'4') % Precomputed
        trainingKernelDataX = [(1:size(trainingDataX,1))' trainingDataX];
        testingKernelDataX = [(1:size(testDataX,1))' testDataX];
    else
        trainingKernelDataX = double(trainingDataX);
        testingKernelDataX = double(testDataX);
    end
    
    tic
    mdl = svmtrain(new_trainingDataY, trainingKernelDataX, buildParam);
    trainingTime = toc;
    
    tic
    [predictY, ~, prob_values] = svmpredict(double(testDataY), testingKernelDataX, mdl);
    testTime = toc;
%     The function 'svmpredict' has three outputs. The first one,
%     predictd_label, is a vector of predicted labels. The second output,
%     accuracy, is a vector including accuracy (for classification), mean
%     squared error, and squared correlation coefficient (for regression).
%     The third is a matrix containing decision values or probability
%     estimates (if '-b 1' is specified). If k is the number of classes, for decision values, 
%     each row includes results of predicting k(k-1/2) binary-class SVMs. For probabilities, 
%     each row contains k values indicating the probability that the testing instance is in
%     each class. Note that the order of classes here is the same as 'Label'
%     field in the model structure.

    predict_score = predictY;

    %     Convert predictY back to actual label
    predictY = class_name(round(predictY));
    
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    label_mat = table(testFileNames, testDataY, predictY, prob_values, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'SVM_scores', 'predict_score'});
%     [CM, GORDER] = confusionmat(testDataY,predictY);
    
end

function param = checkParam(param, paramName)
    if isnumeric(param)
        param = num2str(param);
    end
    if strcmp(param, '') == 0
        param = ['-' paramName ' ' param ' '];
    end
end

function [T,predictT] = prepareLabel(T,predictT)
    T(T==-1) = 0;
    predictT(predictT==-1) = 0;
end

function AUC = plotRocCurve(T,scores)
    % Area Under ROC Curve
    [xAUC,yAUC,tAUC,AUC,OPTROCPT] = perfcurve(T,scores,1);
end

