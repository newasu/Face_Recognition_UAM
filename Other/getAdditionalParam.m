function paramVal = getAdditionalParam( paramIndex, paramString, defaultValue )
%GETADDITIONALPARAM Summary of this function goes here
%   Detailed explanation goes here

    paramVal  = find(strcmp(paramIndex, paramString));
    if isempty(paramVal)
        paramVal = defaultValue;
    else
        paramVal = paramString{paramVal+1};
    end
    
end

