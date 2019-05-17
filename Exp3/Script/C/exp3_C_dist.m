% exp3 C classify Is that same person by Euclidean

clear all

% Settings
experiment_name = 'Exp3';
sub_experiment_name = 'C';
method_name = 'dist';

% Distance's parameter
classify_threshold = 0.85;

numb_run = 1;
numb_cv = 2;
training_sample_percent = 0.001; % percentages of training sample
selected_pose_numb = 3; % number of image used each user
number_comparison = 1;
do_balance_class = 1;

% Save path
default_data_store_path = pwd;
idcs = strfind(pwd,filesep);
default_data_store_path = [default_data_store_path(1:idcs(end)-1) ...
    filesep 'Face_Recognition_UAM_data_store'];
saveFolderPath = {'Result', experiment_name, [experiment_name '_' sub_experiment_name]};
filename = [saveFolderPath{end} '_' method_name];
save_path = MakeChainFolder(saveFolderPath, 'target_path', default_data_store_path);

% %Load data
[diveface_feature, diveface_label] = LoadDiveFaceFull();

% Remove user containing less than contain_image
contain_image = 3;
id = unique(diveface_label.id);
for i = 1 : numel(id)
    temp = find(diveface_label.id == id(i));
    if numel(temp) < contain_image
        diveface_label(temp,:) = [];
    end
end
clear contain_image id i temp

for random_seed = 1 : numb_run
    % Split dataset
    [training_id_index, test_id_index, data, user_index] = SplitDataset(diveface_label,...
        'training_sample_percent', training_sample_percent, 'random_seed', random_seed);
    
    [training_pair_list, training_label_pair_list] = PairDataEachClass(...
        training_id_index, diveface_label,...
        'random_seed', random_seed, 'number_comparison', number_comparison, ...
        'selected_pose_numb', selected_pose_numb);
    
    % Subsampling to balance training set
    if do_balance_class
        [training_pair_list, training_label_pair_list] = BalanceClasses(...
            training_pair_list, training_label_pair_list, 'random_seed', random_seed);
    end

    % Training data
    trainingDataX_1 = diveface_feature(training_pair_list(:,1),:);
    trainingDataX_2 = diveface_feature(training_pair_list(:,2),:);
    training_data_id = training_pair_list;
    trainingDataY = training_label_pair_list;
    
    class_name = categorical(categories(trainingDataY));
    
    % Genarate k-fold indices
    [ kFoldIdx, ~ ] = GetKFoldIndices( numb_cv, trainingDataY, random_seed );
    
    foldLog = [];
    for fold = 1 : numb_cv
        % test the rest part
        testData_1 = trainingDataX_1(kFoldIdx(fold,:),:);
        testData_2 = trainingDataX_2(kFoldIdx(fold,:),:);
        testLabel = trainingDataY(kFoldIdx(fold,:),:);
        testFileNames = training_data_id(kFoldIdx(fold,:),:);
        
        trainingTime = 0;
        
        tic;
        for ii = 1 : numel(testLabel)
            temp(ii) = 1/(1+pdist2(testData_1(ii,:), testData_2(ii,:), 'euclidean'));
        end
        testTime = toc;
        
        filenames = training_data_id;
        labels = trainingDataY;
        same_idx = find(temp >= classify_threshold);
        predict_labels = repmat(class_name(1), numel(trainingDataY), 1);
        predict_labels(same_idx) = class_name(2);
        
        [~,score,~] = my_confusion.getMatrix(double(labels),double(predict_labels),0);

        label_mat = table(filenames, labels, predict_labels, ...
            'variablenames', {'filenames', 'labels', 'predict_labels'});
        
        foldLog = [foldLog; fold score.Accuracy {label_mat} trainingTime testTime];
    end
    
    foldLog = array2table(foldLog, 'VariableNames', {'fold', 'score', ...
         'label_mat', 'trainingTime', 'testTime'});
     
     % Average 5 folds into 1
    avgFoldLog = cell2mat(table2array(foldLog(:, [2 4 5])));
    avgFoldLog = reshape(avgFoldLog',[(size(avgFoldLog,2)), numb_cv, (size(avgFoldLog,1)/numb_cv)]);
    avgFoldLog = sum(avgFoldLog,2)/numb_cv;
    avgFoldLog = reshape(avgFoldLog,[size(avgFoldLog,1) size(avgFoldLog,3)])';
    avgFoldLog = array2table(avgFoldLog, 'VariableNames', {'score', 'trainingTime', 'testTime'});
    
    data_log(random_seed,:) = {foldLog avgFoldLog};
    data_log = cell2table(data_log, 'variablenames', {'foldLog' 'avgFoldLog'});

    % Save
    my_filename = [filename '_' erase(num2str(training_sample_percent),".")];
    my_save_path = [save_path filesep my_filename];
    eval([my_filename ' = data_log;']);
    save(my_save_path, my_filename,'-v7.3');
    eval(['clear ' my_filename])
    clear data_log
%     
end

