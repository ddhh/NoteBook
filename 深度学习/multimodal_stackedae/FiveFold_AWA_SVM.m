
load 'rgsift_tx.mat'
load 'rgsift_ty.mat'
load 'rgsift_trx.mat'
load 'rgsift_try.mat'


[maxd trainlabel] = max(rgsift_try);
[maxd testlabel] = max(rgsift_ty);

nus_train_label=trainlabel';
nus_train_data=rgsift_trx';
%[nus_train_data aaa sss] = featureNormalize(rgsift_trx');
nus_test_label=testlabel';
nus_test_data=rgsift_tx';
%[nus_test_data aa ss] = featureNormalize(rgsift_tx');

model=svmtrain(nus_train_label,nus_train_data);
[nus_predict_label,nus_accuracy,e]=svmpredict(nus_test_label,nus_test_data,model,'-q');
nus_accuracy
