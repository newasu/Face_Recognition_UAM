function my_path = MakeChainFolder(folder_name, varargin)
%MakeChainFolder Summary of this function goes here
%   Detailed explanation goes here
% MakeChainFolder({'Result', 'Exp1', 'Exp1_sub1'}, 'target_path', pwd);

    target_path = getAdditionalParam('target_path', varargin, pwd);

    disp(['Target path: ' target_path]);
    
    my_path = target_path;
    % Make folder if not exist
    for i = 1 : numel(folder_name)
        my_path = [my_path '/' folder_name{i}];
        if ~exist(my_path,'dir')
            mkdir(my_path);
            disp(['Make folder: ' my_path]);
        end
    end
    
end

