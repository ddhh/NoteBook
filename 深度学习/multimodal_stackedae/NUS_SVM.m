
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_WT.mat'
load 'D:\ML Matlab\Data\NUS-WIDE\new_train_WT.mat'
load 'D:\ML Matlab\Data\NUS-WIDE\NWO_Train_Labels.mat'
load 'D:\ML Matlab\Data\NUS-WIDE\NWO_TestLabel.mat'


[maxd trainlabel] = max(NWO_Train_Labels');
[maxd testlabel] = max(BB');

nus_train_label=trainlabel';
nus_train_data=new_train_WT;
nus_test_label=testlabel';
nus_test_data=new_test_WT;

model=svmtrain(nus_train_label,nus_train_data);
[nus_predict_label,nus_accuracy,e]=svmpredict(nus_test_label,nus_test_data,model,'-q');
nus_accuracy