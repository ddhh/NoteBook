%% CS294A/CS294W Softmax Exercise 


clear all;
inputSize = 225 + 64 + 144 + 128 + 73; % Size of input vector (MNIST images are 28x28)
numClasses = 31;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-2; % Weight decay parameter

%%======================================================================
%% STEP 1: Load data

load 'D:\ML Matlab\Data\NUS-WIDE\new_train_CM55.mat'  
load 'D:\ML Matlab\Data\NUS-WIDE\new_train_CH.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_train_CORR.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_train_WT.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_train_EDH.mat'    

load 'D:\ML Matlab\Data\NUS-WIDE\NWO_Train_Labels.mat'
load 'D:\ML Matlab\Data\NUS-WIDE\NWO_TestLabel.mat'

[m1 mu sigma] = featureNormalize(new_train_CM55');
[m2 mu sigma] = featureNormalize(new_train_CH');
[m3 mu sigma] = featureNormalize(new_train_CORR');
[m4 mu sigma] = featureNormalize(new_train_WT');
[m5 mu sigma] = featureNormalize(new_train_EDH');

inputData = [ new_train_CH new_train_CORR new_train_WT new_train_CM55 new_train_EDH];
%images = [m1;m2;m3;m4;m5];


[maxd trainlabel] = max(NWO_Train_Labels');
[maxd testlabel] = max(BB');

nus_train_label=trainlabel';
nus_train_data=inputData;
nus_test_label=testlabel';

size(nus_train_data)

model=svmtrain(nus_train_label,nus_train_data);



%%======================================================================
%% STEP 5: Testing

load 'D:\ML Matlab\Data\NUS-WIDE\new_test_CM55.mat'  
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_CH.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_CORR.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_WT.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_EDH.mat'  

images = [ new_test_CH new_test_CORR new_test_WT new_test_CM55 new_test_EDH];

%[inputData mu sigma] = featureNormalize(images);


nus_train_label=trainlabel';
nus_train_data=new_train_WT;
nus_test_label=testlabel';
nus_test_data=images;

[nus_predict_label,nus_accuracy,e]=svmpredict(nus_test_label,nus_test_data,model,'-q');
nus_accuracy



% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);
[maxy ylabel]= max(labels);
acc = mean(ylabel(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);


