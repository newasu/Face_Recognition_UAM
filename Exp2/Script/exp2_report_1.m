function [result] = exp2_report_1(dataset_label, score)
%EXP2_REPORT_1 Summary of this function goes here
%   Detailed explanation goes here
% [B_welm] = exp2_report_1(diveface_label, Exp2_B_welm.testResult(1,:).scores{1,1});

    id = dataset_label.id;
    pose = dataset_label.pose;
    filename = dataset_label.filename;
    
    gender_cat = categorical(categories(dataset_label.gender));
    ethnicity_cat = categorical(categories(dataset_label.ethnicity));
    
    true_label = [];
    for i = 1 : size(score,1)
        index = find(strcmp(dataset_label.filename, score(i,:).filenames));
        true_label = [true_label; dataset_label(index, {'gender', 'ethnicity'})];
    end
    
    myscore = [score true_label];
    
    temp_result = [];
    temp_cat = [];
    for i = 1 : numel(ethnicity_cat)
        for j = 1 : numel(gender_cat)
            index = find((myscore.ethnicity == ethnicity_cat(i)) &...
                (myscore.gender == gender_cat(j)));
            temp_score = myscore(index,:);
            temp_result = [ temp_result; (sum(temp_score.labels == ...
                temp_score.predict_labels)/size(temp_score,1))];
            temp_cat = [temp_cat; ...
                strcat(cellstr(gender_cat(j)),'_',cellstr(ethnicity_cat(i)))];
        end
    end
    
    result = table(temp_cat, temp_result, 'variablenames', {'class' 'score'});
end

