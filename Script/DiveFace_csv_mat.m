
folder_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu''s Mac/PhD''s Degree/New/SourceCode/Face_Recognition_UAM_data_store/Dataset/DiveFace_ArcFace/';

df_data = [];
df_name = [];

for i = 1 : 7
    % Prepare data
    temp_data = csvread([folder_path 'df_fe_' num2str(i) '.csv']);
    temp_name = readtable([folder_path 'df_fn_' num2str(i) '.txt'], 'ReadVariableNames',false);
    temp_name_idx = cellfun(@(x) strfind(x, '.'), temp_name.Var2, 'UniformOutput', false);
    temp_name = cellfun(@(x,y) x(1:y-1), temp_name.Var2, temp_name_idx, 'UniformOutput', false);
    temp_name = str2double(temp_name);

    % Sort
    [~, b] = sort(temp_name);
    temp_data = temp_data(b, :);
    temp_name = temp_name(b);

    df_data = [df_data; temp_data];
    df_name = [df_name; temp_name];
    
    disp([num2str(i) ' finished']);
end

[~, diveface_label] = LoadDiveFaceFull();
diveface_feature = df_data;

clear temp_data temp_name temp_name_idx b i
clear folder_path df_data df_name

save('DiveFace_ArcFace');



