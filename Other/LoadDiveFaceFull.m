function [raw_feature, label] = LoadDiveFaceFull(varargin)
%LOADDIVEFACEFULL Summary of this function goes here
%   Detailed explanation goes here

    % set dafault path if not specific
    root_data_store_path = getAdditionalParam( 'dataset_path', varargin, pwd );
    idcs   = strfind(pwd,filesep);
    root_data_store_path = [root_data_store_path(1:idcs(end)-1) ...
        filesep 'Face_Recognition_UAM_data_store' filesep 'Dataset' filesep 'DiveFace'];
    
    dataset_path = getAdditionalParam( 'dataset_path', varargin, root_data_store_path );
    network_type = getAdditionalParam( 'network_type', varargin, 'ResNet' ); % ResNet VGG ArcFace
    
    read_type = getAdditionalParam( 'read_type', varargin, 'mat' );
    
    diveface_path = [dataset_path '_' network_type filesep];
    
    disp(['Dataset path: ' diveface_path]);
    disp('Importing DiveFace..');
    
    if strcmp(read_type, 'mat')
        data = load([diveface_path 'DiveFace_' network_type]);
        raw_feature = data.diveface_feature;
        label = data.diveface_label;
        
    else
        % Each feature vector is composed by 2048 featutes
        raw_feature = csvread([diveface_path 'embeddings_full_dataset_' network_type '.csv']);

        %column 1= Gender (0 Male, 1 Female), column 2= Ethnicity (0 Black, 1 Asian, 2 Caucasian) column 3= ID
        raw_label = csvread([diveface_path 'attributes_full_dataset.csv']); 
        
        % Read file name labels
%         raw_label_filename = importdata([diveface_path 'files.txt']);
        temp = strfind(dataset_path, filesep);
        temp = [dataset_path(1:temp(end)) 'label_filenames.txt'];
        fileID = fopen(temp);
        raw_filename = textscan(fileID,'%s');
        fclose(fileID);
        raw_filename = raw_filename{1};
        [raw_filepath, raw_filename, raw_fileextension] = cellfun(@(x) ...
            fileparts(x), raw_filename, 'UniformOutput', false);
        clear temp fileID

    %     Remove row and column header
        raw_feature(1,:) = [];
        raw_feature(:,1) = [];
        raw_label(1,:) = [];
        raw_label(:,1) = [];

        gender_index = 1;
        ethnicity_index = 2;
        id_index = 3;
        pose_index = 4;
        data_id_index = 5;

        gender_cat = categorical({'male', 'female'});
        ethnicity_cat = categorical({'black', 'asian', 'caucasian'});

        % Sort by user id and pose id
        [my_data_label, sortIndex] = sortrows(raw_label, id_index);
        raw_feature = raw_feature(sortIndex, :);
        raw_filepath = raw_filepath(sortIndex);
        raw_filename = raw_filename(sortIndex);
        raw_fileextension = raw_fileextension(sortIndex);
        id = unique(my_data_label(:, id_index));
        my_data_label = [my_data_label zeros(size(raw_label,1),1)];
        for i = 1 : numel(id)
            disp(['DiveFace is finding user ID ' num2str(id(i))]);
            temp = find(raw_label(:, id_index) == id(i));
            my_data_label(temp,end) = 1:numel(temp);
        end

        gender = gender_cat(my_data_label(:,gender_index) + 1)';
        ethnicity = ethnicity_cat(my_data_label(:,ethnicity_index) + 1)';
        data_id = 1:numel(gender);

    %     For temporarily
        raw_label_filename = strcat(string(my_data_label(:, id_index)), '_',...
            string(my_data_label(:, pose_index)));

        % Bind into table
        label = table(gender, ethnicity, my_data_label(:, id_index),...
            my_data_label(:, pose_index), data_id', raw_filepath, raw_filename, raw_fileextension,...
            'VariableNames', {'gender', 'ethnicity', 'id', 'pose', 'data_id', 'filepath', 'filename', 'fileext'});
    end
    
    disp('Loaded DiveFace..');
end

