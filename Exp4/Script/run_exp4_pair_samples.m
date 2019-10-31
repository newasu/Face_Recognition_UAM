
clear

% pair samples

experiment_name = 'Exp4';
number_subdataset = 20; % 10 20 50 100
do_experiment = 1:number_subdataset; % run experiment number

% %Load data
[diveface_feature, diveface_label] = LoadDiveFaceFull();
diveface_data_id = diveface_label.data_id;

% Remove user containing less than contain_image
disp('Removing error users..');
user_contain_image = 3;
id = unique(diveface_label.id);
for i = 1 : numel(id)
    temp = find(diveface_label.id == id(i));
    if numel(temp) < user_contain_image
        disp('Deleting user data_id: ');
        num2str(diveface_data_id(temp))
        diveface_label(temp,:) = [];
        diveface_feature(temp,:) = [];
        diveface_data_id(temp) = [];
    end
end
clear contain_image id i temp
disp('Removed error users');

% Generate training/test set
[subdataset, subdataset_label] = SplitSubdataset(diveface_label.id, ...
    {diveface_label.gender, diveface_label.ethnicity}, ...
    'number_sub_dataset', number_subdataset);
concat_subdataset_label = strcat(subdataset_label(:,1), '_', subdataset_label(:,2));

for experiment_round = do_experiment
    
    try
        
        % Check save path
        default_data_store_path = pwd;
        idcs = strfind(pwd,filesep);
        default_data_store_path = [default_data_store_path(1:idcs(end)-1) ...
            filesep 'Face_Recognition_UAM_data_store'];
        paired_list_save_path = MakeChainFolder({'Other', experiment_name, ...
            [experiment_name '_' num2str(number_subdataset)]}, 'target_path', default_data_store_path);
        
        % Prepare data
        disp('Preparing data..');

        % Assign training/test set
        training_set_sample_idx = [];
        test_set_sample_idx = [];
        for i = 1 : number_subdataset % Train one and test nine
            if experiment_round == i % training set
                training_set_sample_idx = subdataset{i};
            else % test set
                test_set_sample_idx = [test_set_sample_idx; subdataset{i}];
            end
        end
        
        % stopwatch
        running_time = tic;
            
        disp('Generateing training_paired_label..');
        temp_paired_list_save_path = [paired_list_save_path filesep ...
            experiment_name '_' num2str(number_subdataset) '_training_paired_label_' num2str(experiment_round)];
        training_paired_label = PairSampleClassesEqually(...
            training_set_sample_idx, subdataset_label, diveface_label,...
            'random_seed', experiment_round);
        save(temp_paired_list_save_path, 'training_paired_label', '-v7.3');

        disp('Generateing test_paired_label..');
        temp_paired_list_save_path = [paired_list_save_path filesep ...
            experiment_name '_' num2str(number_subdataset) '_test_paired_label_' num2str(experiment_round)];
        test_paired_label = PairSampleClassesEqually(...
                test_set_sample_idx, subdataset_label, diveface_label,...
                'random_seed', experiment_round);
        save(temp_paired_list_save_path, 'test_paired_label', '-v7.3');
        
    catch ME
        MailNotify('Subject', ['sample pairing ' num2str(experiment_round) '/' ... 
            num2str(number_subdataset) ' has got ERROR'], ...
        	'Message', [ ME.message ' at ' char(datetime('now','TimeZone','+07:00'))]);
        
    end
    
    % stopwatch
    running_time = toc(running_time);
    
    MailNotify('Subject', ['Sample pairing ' num2str(experiment_round) '/' ...
        num2str(number_subdataset) ' finished'], ...
        'Message', ['Sample pairing ' num2str(experiment_round) '/' ...
        num2str(number_subdataset) ' has been finished at ' ...
        char(datetime('now','TimeZone','+07:00')) '. (' num2str(running_time) ' seconds)']);
    
end


