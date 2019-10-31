function [biometric_perf_threshold, biometric_perf_mat] = exp3_report_biometric_perf(...
    true_label, predict_label, score, pos_class, varargin)
%EXP3_REPORT_BIOMETRIC_PERF Summary of this function goes here
%   Detailed explanation goes here

    % FAR (False Acceptance Rate) = FMR (False Match Rate) = should reject but accpet
    % FRR (False Rejection Rate) = FNMR (False Non-Match Rate) = should accpet but reject
    % EER = Equal Error Rate = crossing point between FAR meet FRR
    
    positive_class_score_order = getAdditionalParam( 'positive_class_score_order', varargin, 'ascend');
    threshold_step = getAdditionalParam( 'threshold_step', varargin, 0.0001 );
    threshold_range = getAdditionalParam( 'threshold_range', varargin, [min(score), max(score)] );
    doPlot = getAdditionalParam( 'doPlot', varargin, 0);
    
    % declare limit of threshold
    [~, n] = getprecision(threshold_step);
    threshold_range(1) = floor(threshold_range(1) * 10^n)/10^n;
    threshold_range(2) = ceil(threshold_range(2) * 10^n)/10^n;
    threshold = threshold_range(1): threshold_step : threshold_range(2);
    
    % AUC
    if strcmp(positive_class_score_order, 'ascend')
    	[x,y,~,auc,~,~,~] = perfcurve(true_label, score, pos_class);
    else
        [x,y,~,auc,~,~,~] = perfcurve(true_label, 1./score, pos_class);
    end
    
    % EER
    eer = InterX([x'; y'], [0,1; 1,0]);
    eer = eer(1);

    pos_idx = true_label==pos_class;
    neg_idx = ~pos_idx;
    
    if ~strcmp(positive_class_score_order, 'ascend')
        threshold = flip(threshold);
    end
    
    % FAR, FRR, EER curve
    far = [];
    frr = [];
    for i = 1 : numel(threshold)
        if strcmp(positive_class_score_order, 'ascend')
            temp_threshold_class = score >= threshold(i);
        else
            temp_threshold_class = score < threshold(i);
        end
        
        % FAR
        temp_far = sum(temp_threshold_class(neg_idx) == 1)/sum(neg_idx);
        far = [far; temp_far];
        
        % FRR
        temp_frr = sum(temp_threshold_class(pos_idx) == 0)/sum(pos_idx);
        frr = [frr; temp_frr];

    end
    
    % False Non-Match Rate
    temp_FMR_0d1 = find(far < 0.001);
    if ~isempty(temp_FMR_0d1)
        temp_FMR_0d1 = frr(temp_FMR_0d1(1)) * 100;
    else
        temp_FMR_0d1 = nan;
    end
    
    temp_FMR_0d01 = find(far < 0.0001);
    if ~isempty(temp_FMR_0d01)
        temp_FMR_0d01 = frr(temp_FMR_0d01(1)) * 100;
    else
        temp_FMR_0d01 = nan;
    end
    
    if ~strcmp(positive_class_score_order, 'ascend')
        threshold = flip(threshold);
        far = flip(far);
        frr = flip(frr);
    end
    
    % Genaral performance matrix
    [TP, FP, TN, FN] = calError((true_label==pos_class), (predict_label==pos_class));
    N = TP + FP + TN + FN;
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    f1_score = 2*((precision * recall)/(precision + recall));
    PPV = TP/(TP+FP);
    
    biometric_perf_threshold = table(threshold', far, frr, ...
        'variablenames', {'threshold' 'far' 'frr'});
    
    biometric_perf_mat = table((TP+TN)/N, auc, f1_score, ...
        precision, recall, PPV, ...
        eer, temp_FMR_0d1, temp_FMR_0d01, ...
        'variablenames', {'accuracy', 'AUC', 'F1', ...
        'precision', 'recall', 'PPV', ...
        'EER', 'FMR_0d1', 'FMR_0d01'});
    
    % Plot
    if doPlot
    %     Plot AUC
        figure();
        plot(biometric_perf_threshold.far, 1-biometric_perf_threshold.frr);
        title('AUC');

    %     Plot EER
        figure();
        plot(biometric_perf_threshold.threshold, biometric_perf_threshold.far);
        hold on;
        plot(biometric_perf_threshold.threshold, biometric_perf_threshold.frr);
        hold off;
        title('EER');

        figure();
        plot(biometric_perf_threshold.far, biometric_perf_threshold.frr)
        title('FAR vs FRR');
    end
end

function [p, n] = getprecision(x)
    f = 14-9*isa(x,'single'); % double/single = 14/5 decimal places.
    s = sprintf('%.*e',f,x);
    v = [f+2:-1:3,1];
    s(v) = '0'+diff([0,cumsum(s(v)~='0')>0]);
    p = str2double(s);
    
    x = abs(x); %in case of negative numbers
    n = 0;
    while (floor(x*10^n)~=x*10^n)
        n = n + 1;
    end
end

