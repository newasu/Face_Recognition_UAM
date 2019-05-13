function SetRandomSeed(randomseed)
%SETRANDOMSEED Summary of this function goes here
%   Detailed explanation goes here

    if randomseed ~= 0
        rng(randomseed);
    end
end

