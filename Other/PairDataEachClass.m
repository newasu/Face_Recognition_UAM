function [pair_list, label_pair_list] = PairDataEachClass(data, data_info, varargin)
%PAIRDATAEACHCLASS Summary of this function goes here
%   Detailed explanation goes here

    number_comparison = getAdditionalParam('number_comparison', varargin, 1);
    selected_pose_numb = getAdditionalParam('selected_pose_numb', varargin, 2);
    random_seed = getAdditionalParam('random_seed', varargin, 1); % set to 0 for freely random
    
%     data_varnames = categorical(data.Properties.VariableNames);
%     data_rownames = categorical(data.Properties.RowNames);

    my_label = categorical({'same', 'different'});
    data_size = cell2mat(cellfun(@(x) size(x,1), table2cell(data), 'UniformOutput', false));
    total_round = num2str((numel(data) * selected_pose_numb * number_comparison * ...
        sum(sum(data_size)))+(sum(sum(data_size)) * selected_pose_numb));
    
    global my_random_seed
    my_random_seed = random_seed;
    
    temp_pair_list = [];
    temp_label_pair_list = [];
    pair_flag = [0, 0];
    
    % Pair
    my_data = table2cell(data);
    for i = 1 : numel(my_data)
        temp_my_data = my_data{i};
        
        for j = 1 : numel(temp_my_data)
            % Find data
            temp_data_info = find(data_info.id == temp_my_data(j));
            temp_selected_pose_numb = min(selected_pose_numb, size(temp_data_info,1));
            temp_data_info = data_info(temp_data_info(1:temp_selected_pose_numb),:);
            
            % Pair same group
            pair_same_group = nchoosek(temp_data_info.data_id, 2);
            temp_pair_list = [temp_pair_list; pair_same_group];
            temp_label_pair_list = [temp_label_pair_list; ones(size(pair_same_group,1),1)];
            
            % Pair other group
            for k = 1 : size(temp_data_info,1)
                
                % Random in each class
                for l = 1 : numel(my_data)
                    temp_pair_my_data = my_data{l};
                    
                    for m = 1 : number_comparison
                        temp_repeated_random = [0, 0];
                        while 1
                            % Random pair
                            while 1
                                rand_data = myRandom(numel(temp_pair_my_data));
                                temp_selected_pose_numb = size(...
                                    data_info(find(data_info.id ==...
                                    temp_pair_my_data(rand_data(1))),:),1);
                                rand_sub_data = myRandom(temp_selected_pose_numb);
                                % Avoid repeated random member
                                if ~ismember([rand_data(1), rand_sub_data(1)],...
                                        temp_repeated_random, 'rows')
                                    break;
                                else
                                    disp('repeated pairing compared in sub set')
                                end
                            end
                            
                            temp_pair_data_info = data_info(...
                                find(data_info.id == temp_pair_my_data(rand_data(1)) &...
                                data_info.pose == rand_sub_data(1)), :);

                            temp_pair = [temp_data_info.data_id(k) temp_pair_data_info.data_id];

                            % Check repeat for whole set
                            if ~ismember(temp_pair, pair_flag, 'rows') &...
                                    ~ismember(flip(temp_pair), pair_flag, 'rows') &...
                                    ~ismember(temp_data_info.data_id(k), temp_pair_data_info.data_id)
                                    [pair_flag, temp_pair_list, temp_label_pair_list] = add_pair(...
                                        pair_flag,temp_pair, temp_pair_list, temp_label_pair_list);
                                    disp(['Paired: ' num2str(size(temp_pair_list,1)) '/' total_round]);
                                break;
                            else
                                temp_repeated_random = [temp_repeated_random;...
                                    rand_data(1) rand_sub_data(1)];
                                disp('repeated pairing compared in whole set')
                            end
                        end
                    end
                end
            end
        end
    end

    pair_list = temp_pair_list;
    label_pair_list = my_label(temp_label_pair_list)';
    
    clear global my_random_seed
end

function rn = myRandom(elementNumb)
    global my_random_seed
    SetRandomSeed(my_random_seed);
    rn = randperm(elementNumb);
    my_random_seed = my_random_seed + 1;
end

function [pf, tpl, tlpl] = add_pair(pf, tp, tpl, tlpl)
    pf = [pf; tp];
    tpl = [tpl; tp];
    tlpl = [tlpl; 2];
end

