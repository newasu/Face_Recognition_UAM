function [paired_labels] = PairSampleClassesEqually(data, data_label, dataset_label, varargin)
%PAIRSAMPLECLASSESEQUALLY Summary of this function goes here
%   Detailed explanation goes here

    random_seed = getAdditionalParam('random_seed', varargin, 1); % set to 0 to not consider
    number_positive_each_sample = getAdditionalParam('number_positive_each_sample', varargin, 3);
    
    my_label = categorical({'same', 'different'});
    data_size = numel(data);
    total_round = num2str(data_size * (number_positive_each_sample + ...
        (number_positive_each_sample * size(data_label,1))));
    
    % Prepare data
    sample_id = repmat(data(:), 1, 3)';
    sample_id = reshape(sample_id, (size(data,1) * number_positive_each_sample),  size(data_label,1));
    sample_pose = 1 : number_positive_each_sample;
    sample_pose = repmat(sample_pose, 1, data_size);
    sample_pose = reshape(sample_pose, (size(data,1) * number_positive_each_sample),  size(data,2));
    sample_pose(isnan(sample_id)) = nan; % assign nan to error users

    % shuffle samples
    my_random_seed = random_seed;
    for ii = 1 : size(data_label,1)
        temp = myRandom(size(sample_id,1), my_random_seed);
        sample_id(:,ii) = sample_id(temp,ii);
        sample_pose(:,ii) = sample_pose(temp,ii);
        my_random_seed = my_random_seed + 1;
    end

    temp_pair = [];
    temp_pair_label = [];
    temp_id = [];
    temp_pose = [];
    temp_add_order = [];

    % Pair class same
    disp('Pairing class of same..');
    for ii = 1 : data_size
        % consider user
        consider_data = data(ii);
        if ~isnan(consider_data)
            % find considered user in dataset
            temp_idx = find(dataset_label.id == consider_data);
            % find data_id of user
            consider_data_in_table = dataset_label.data_id(temp_idx);
            temp_temp = nchoosek(consider_data_in_table, 2);
            
            % bind into list
            for jj = 1 : numel(consider_data_in_table)
                temp_pair = [temp_pair; temp_temp(jj,:)];
                temp = [my_label(1) ...
                    dataset_label.gender(temp_idx(jj)) dataset_label.ethnicity(temp_idx(jj)) ...
                    dataset_label.gender(temp_idx(jj)) dataset_label.ethnicity(temp_idx(jj))];
                temp_pair_label = [temp_pair_label; temp];
                temp_id = [temp_id; consider_data consider_data];
                temp_pose = [temp_pose; ...
                    dataset_label.pose(dataset_label.data_id == temp_temp(jj,1)) ...
                    dataset_label.pose(dataset_label.data_id == temp_temp(jj,2))];
                temp_add_order(end+1) = numel(temp_add_order) + 1;
                
                disp(['Paired: ' num2str(numel(temp_add_order)) '/' total_round]);
            end
            
%             temp_pair = [temp_pair; temp];
%             temp = [repmat(my_label(1), 3, 1) ...
%                 dataset_label.gender(temp_idx) dataset_label.ethnicity(temp_idx) ...
%                 dataset_label.gender(temp_idx) dataset_label.ethnicity(temp_idx)];
%             temp_pair_label = [temp_pair_label; temp];
%             temp_id = [temp_id; repmat(consider_data, 3, 1) repmat(consider_data, 3, 1)];
        end
    end
    clear consider_data consider_data_in_table temp_idx temp
    disp('Class of same were paired.');
    
    % Pair class different
    disp('Pairing class of different..');
    for xx = 1 : size(data,2)
        for yy = 1 : size(data,1)
            
            % consider user
            consider_data = data(yy,xx);
            if ~isnan(consider_data)
                
                % find considered user in dataset
                temp_idx = find(dataset_label.id == consider_data);
                
                % find data_id of user
                consider_data_in_table = dataset_label.data_id(temp_idx);
                
                % avoid pairing samples within same class
                temp_sample_id = sample_id;
                temp_sample_id(temp_sample_id==consider_data) = 0;
                
                % loop for pairing every class
                for ii = 1 : size(data_label,1)
                    temp_temp_sample_id = temp_sample_id(:,ii);
                    temp_temp_sample_pose = sample_pose(:,ii);
                    temp_temp_idx = 1;
                    
                    % loop for each pose of considered user
                    for jj = 1 : number_positive_each_sample
                        temp_flag = 1;
                        
                        % check condition
                        while temp_flag
                            if temp_temp_sample_id(temp_temp_idx) == 0 % if 0 skip same class
                               temp_temp_idx = temp_temp_idx + 1;
                               disp(['Paired: skip same class']);
                            elseif isnan(temp_temp_sample_id(temp_temp_idx)) % if nan move NaN to last element of vector
                                temp_temp_sample_id(temp_temp_idx) = [];
                                temp_temp_sample_id(end+1) = nan;
                                temp_temp_sample_pose(temp_temp_idx) = [];
                                temp_temp_sample_pose(end+1) = nan;
                                disp(['Paired: skip error user']);
                            else % bind paired sample into list
                                % find paired sample in dataset
                                temp = dataset_label.id == temp_temp_sample_id(temp_temp_idx);
                                temp = dataset_label(temp,:);
                                temp = temp(temp.pose == temp_temp_sample_pose(temp_temp_idx),:);
                                temp_temp_pair = [consider_data_in_table(jj) temp.data_id];
                                
                                % check repeated pairing
                                if ismember(temp_temp_pair, temp_pair, 'rows') || ismember(flip(temp_temp_pair), temp_pair, 'rows')
                                    % skip element if it was being paired with this data_id
                                    temp_temp_idx = temp_temp_idx + 1;
                                    disp(['Paired: skip repeated pairing']);
                                else % bind
                                    temp_pair = [temp_pair; temp_temp_pair];
                                    temp = [my_label(2) ...
                                        dataset_label.gender(temp_idx(jj)) ...
                                        dataset_label.ethnicity(temp_idx(jj)) ...
                                        temp.gender ...
                                        temp.ethnicity];
                                    temp_pair_label = [temp_pair_label; temp];
                                    temp_id = [temp_id; consider_data temp_temp_sample_id(temp_temp_idx)];
                                    temp_pose = [temp_pose; ...
                                        dataset_label.pose(dataset_label.data_id == consider_data_in_table(jj)) ...
                                        temp_temp_sample_pose(temp_temp_idx)];
                                    temp_add_order(end+1) = numel(temp_add_order) + 1;
                                    
                                    % move used element to last element in vector
                                    temp_temp_sample_id(end+1) = temp_temp_sample_id(temp_temp_idx);
                                    temp_temp_sample_id(temp_temp_idx) = [];
                                    temp_temp_sample_pose(end+1) = temp_temp_sample_pose(temp_temp_idx);
                                    temp_temp_sample_pose(temp_temp_idx) = [];
                                    
                                    % stop while
                                    temp_flag = 0;
                                    
                                    disp(['Paired: ' num2str(size(temp_pair,1)) '/' total_round]);
                                end
                                
                            end
                        end
                    end
                    
                    temp_temp_sample_id(temp_temp_sample_id==0) = consider_data;
                    sample_id(:,ii) = temp_temp_sample_id;
                end
            end
            
        end
    end

    paired_labels = table(temp_pair_label(:,1), ...
        temp_id(:,1), temp_pair(:,1), temp_pose(:,1), temp_pair_label(:,2), temp_pair_label(:,3),...
        temp_id(:,2), temp_pair(:,2), temp_pose(:,2), temp_pair_label(:,4), temp_pair_label(:,5),...
        temp_add_order', ...
        'variablenames', {'paired_label', ...
        'master_id', 'master_data_id', 'master_pose', 'master_gender', 'master_ethnicity', ...
        'paired_id', 'paired_data_id', 'paired_pose', 'paired_gender', 'paired_ethnicity', ...
        'added_order'});
    
end

function rn = myRandom(elementNumb, seed)
    SetRandomSeed(seed);
    rn = randperm(elementNumb);
end


