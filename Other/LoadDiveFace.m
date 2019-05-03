function [raw_feature, label] = LoadDiveFace(varargin)
%LOADDIVEFACE Summary of this function goes here
%   Detailed explanation goes here

    dataset_path = getAdditionalParam( 'dataset_path', varargin, pwd );
    
    diveface_path = [dataset_path '/Dataset/DiveFace1/'];
    
    %Each feature vector is composed by 2048 featutes
    raw_feature = importdata([diveface_path 'embeddings.txt']);

    %column 1= Gender (0 Male, 1 Female), column 2= Ethnicity (0 Black, 1 Asian, 2 Caucasian) column 3= ID
    raw_label = importdata([diveface_path 'attributes.txt']); 

    raw_label_filename = importdata([diveface_path 'files.txt']);
    
    gender_index = 1;
    ethnicity_index = 2;
    id_index = 3;
    pose_index = 4;
    
    gender_cat = categorical({'male' 'female'});
    ethnicity_cat = categorical({'black' 'asian' 'caucasian'});

    % Sort by user id and pose id
    my_data_label = sortrows(raw_label, id_index);
    id = unique(my_data_label(:, id_index));
    my_data_label = [my_data_label zeros(size(raw_label,1),1)];
    for i = 1 : numel(id)
        temp = find(raw_label(:, id_index) == id(i));
        my_data_label(temp,end) = 1:numel(temp);
    end
    
    gender = gender_cat(my_data_label(:,gender_index) + 1)';
    ethnicity = ethnicity_cat(my_data_label(:,ethnicity_index) + 1)';
    
    % Bind into table
    label = table(gender, ethnicity, my_data_label(:, id_index),...
        my_data_label(:, pose_index), raw_label_filename,...
        'VariableNames', {'gender', 'ethnicity', 'id', 'pose', 'filename'});
    
end

