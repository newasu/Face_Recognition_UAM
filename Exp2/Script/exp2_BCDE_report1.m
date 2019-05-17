
[diveface_feature, diveface_label] = LoadDiveFaceFull();

[B_dist_near] = exp2_report_1(diveface_label, Exp2_B_dist_near.testResult(1,:).scores{1,1});
[B_dist_mean] = exp2_report_1(diveface_label, Exp2_B_dist_mean.testResult(1,:).scores{1,1});
[B_lda] = exp2_report_1(diveface_label, Exp2_B_lda.testResult(1,:).scores{1,1});
[B_mlp] = exp2_report_1(diveface_label, Exp2_B_mlp.testResult(1,:).scores{1,1});
[B_welm] = exp2_report_1(diveface_label, Exp2_B_welm.testResult(1,:).scores{1,1});
[B_svm] = exp2_report_1(diveface_label, Exp2_B_svm.testResult(1,:).scores{1,1});


[C_dist_near] = exp2_report_1(diveface_label, Exp2_C_dist_near.testResult(1,:).scores{1,1});
[C_dist_mean] = exp2_report_1(diveface_label, Exp2_C_dist_mean.testResult(1,:).scores{1,1});
[C_lda] = exp2_report_1(diveface_label, Exp2_C_lda.testResult(1,:).scores{1,1});
[C_mlp] = exp2_report_1(diveface_label, Exp2_C_mlp.testResult(1,:).scores{1,1});
[C_welm] = exp2_report_1(diveface_label, Exp2_C_welm.testResult(1,:).scores{1,1});
[C_svm] = exp2_report_1(diveface_label, Exp2_C_svm.testResult(1,:).scores{1,1});


[D_dist_near] = exp2_report_1(diveface_label, Exp2_D_dist_near.testResult(1,:).label_mat{1,1});
[D_dist_mean] = exp2_report_1(diveface_label, Exp2_D_dist_mean.testResult(1,:).label_mat{1,1});
[D_lda] = exp2_report_1(diveface_label, Exp2_D_lda.testResult(1,:).label_mat{1,1});
[D_mlp] = exp2_report_1(diveface_label, Exp2_D_mlp.testResult(1,:).label_mat{1,1});
[D_welm] = exp2_report_1(diveface_label, Exp2_D_welm.testResult(1,:).label_mat{1,1});
[D_svm] = exp2_report_1(diveface_label, Exp2_D_svm.testResult(1,:).label_mat{1,1});


[E_dist_near] = exp2_report_1(diveface_label, Exp2_E_dist_near.testResult(1,:).label_mat{1,1});
[E_dist_mean] = exp2_report_1(diveface_label, Exp2_E_dist_mean.testResult(1,:).label_mat{1,1});
[E_lda] = exp2_report_1(diveface_label, Exp2_E_lda.testResult(1,:).label_mat{1,1});
[E_mlp] = exp2_report_1(diveface_label, Exp2_E_mlp.testResult(1,:).label_mat{1,1});
[E_welm] = exp2_report_1(diveface_label, Exp2_E_welm.testResult(1,:).label_mat{1,1});
[E_svm] = exp2_report_1(diveface_label, Exp2_E_svm.testResult(1,:).label_mat{1,1});


