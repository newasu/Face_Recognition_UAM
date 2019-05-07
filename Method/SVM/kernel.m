function [K] = kernel(X_train, X_test, kernelType, varargin)
%KERNEL Summary of this function goes here
%   Detailed explanation goes here

    gamma = getAdditionalParam( 'gamma', varargin, 1/(size(X_train,2)) );

    if strcmp(kernelType, 'linear') % linearKernel
        K = linearKernel(X_train, X_test);
    else
        K = rbfKernel(X_train, X_test, gamma);
    end
    
    K = double(K);
end

function K = linearKernel(x_tr, x_te)
    K = x_tr * x_te';
end

function K = rbfKernel(x_tr, x_te,gamma)
    K = exp(-gamma .* pdist2(x_tr,x_te,'euclidean').^2);
end