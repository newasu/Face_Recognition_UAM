function [accuracy, auc, eer] = exp3_report_auc_eer(true_label, predict_label, score, positive_class)
%EXP3_REPORT_AUC_EER Summary of this function goes here
%   Detailed explanation goes here

    % Accuracy
    accuracy = sum(true_label == predict_label) / numel(true_label);

    % AUC
    [x,y,~,auc,~,~,~] = perfcurve(true_label, score, positive_class);
    
    % EER
    eer = InterX([x'; y'], [0,1; 1,0]);
    eer = eer(1);
    
    % plot AUC
    % plot(X,Y);
    
end

