function [predictY, accuracy, mdl, scores, trainingTime, testTime] = ldaClassify(...
    trainingDataX, trainingDataY, testDataX, testDataY, testFileNames, varargin)
%LDACLASSIFY Summary of this function goes here
%   Detailed explanation goes here

    balance = getAdditionalParam( 'balance', varargin, 1 );
    kernelType = getAdditionalParam( 'kernelType', varargin, 'linear' );
    
    class_name = categorical(categories(trainingDataY));
    [new_trainingDataY, ~] = grp2idx(trainingDataY);
    
    tic
    [W] = trainLDA_onevsall(trainingDataX, new_trainingDataY, balance, kernelType );
    trainingTime = toc;
    
    tic
    [predictY] = testLDA(testDataX, trainingDataX, W, kernelType);
    testTime = toc;
    
%     Convert predictY back to actual label
    predictY = class_name(predictY);
    
    accuracy = (sum(predictY==testDataY)/numel(testDataY)) * 100;
    scores = table(testFileNames, testDataY, predictY, 'VariableNames', {'filenames' 'labels', 'predict_labels'});
    
    mdl = table(W);

end

function [W] = trainLDA_onevsall(X, T, balance, kernelType )
    
    if balance == 1
        [S,SI] = hist(T,unique(T));
        maxS = max(S);
        S = sqrt(S.\maxS);
    else
        SI = unique(T);
        S = ones(1,numel(SI));
    end
    
    X = kernel(X,X, kernelType);
    
    T_onehot = cell2mat(arrayfun(@(x) convert_onehot(x,numel(SI)), T, 'UniformOutput', false));
    W = pinv([ones(size(X,1),1) X]) * T_onehot;

end

function xxx = kernel(xx,yy,kt)
    if strcmp(kt,'linear')
        xxx = xx*yy';
    else
        xxx = xx;
    end
end

function oh = convert_onehot(c,nc)
    oh = zeros(1,nc);
    oh(c) = 1;
end

function [predictY] = testLDA(Xtest, Xtrain, W, kernelType )
    Xtest = kernel(Xtest, Xtrain, kernelType);
    predictY = [ones(size(Xtest,1),1) Xtest] * W;
    [~, predictY] = max(predictY,[],2);
end

