function [predictY, score, mdl, label_mat, trainingTime, testTime] = ldaClassify(...
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
    [predictY, predict_score] = testLDA(testDataX, trainingDataX, W, kernelType);
    testTime = toc;
        
%     Convert predictY back to actual label
    predictY = class_name(predictY);
    
%     score = (sum(predictY==testDataY)/numel(testDataY)) * 100;
    [~,score,~] = my_confusion.getMatrix(double(testDataY),double(predictY),0);
    label_mat = table(testFileNames, testDataY, predictY, predict_score, ...
        'VariableNames', {'filenames' 'labels', 'predict_labels', 'predict_score'});
    
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
    
%     T_onehot = cell2mat(arrayfun(@(x) convert_onehot(x,numel(SI)), T, 'UniformOutput', false));
    T_onehot = full(ind2vec(double(T)')');
    
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

function [predictY, predict_score] = testLDA(Xtest, Xtrain, W, kernelType )
    Xtest = kernel(Xtest, Xtrain, kernelType);
    predictY = [ones(size(Xtest,1),1) Xtest] * W;
    predict_score = predictY;
    [~, predictY] = max(predictY,[],2);
end

