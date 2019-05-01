function SetRandomSeed(randomseed)
%SETRANDOMSEED Summary of this function goes here
%   Detailed explanation goes here

    if randomseed ~= 999
        rng(randomseed);
    end
end

