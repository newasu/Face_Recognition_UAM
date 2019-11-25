
data_path = '/Volumes/WK''s Ext_d/DiveFace4K_Full';
save_path = [pwd '_data_store/temp/'];

[~, diveface_label] = LoadDiveFaceFull();

error_img = [];
for i = 1 : size(diveface_label,1)

    try
        test_img = imread([data_path diveface_label.filepath{i}(2:end) '/' diveface_label.filename{i} diveface_label.fileext{i}]);
        imwrite(test_img, [save_path 'df_' num2str(i) '.jpg']);

        disp(['Wrote image: ' num2str(i)]);
        
    catch ME
        error_img = [error_img i];
        warning(['Error image: ' num2str(i)]);
    end
end