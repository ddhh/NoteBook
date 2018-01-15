clear all;clc

numClasses = 31;

cq_inputSize = 225; % CM55
lss_inputSize = 64; %CH
phog_inputSize = 144; %CORR
sift_inputSize = 128; %WT
surf_inputSize = 73; %EDH

%% hidden layer size

cq_hiddenSizeL1 = 100;    % Layer 1 Hidden Size
cq_hiddenSizeL2 = 20;    % Layer 2 Hidden Size

lss_hiddenSizeL1 = 30;    % Layer 1 Hidden Size
lss_hiddenSizeL2 = 20;    % Layer 2 Hidden Size

phog_hiddenSizeL1 = 80;    % Layer 1 Hidden Size
phog_hiddenSizeL2 = 20;    % Layer 2 Hidden Size

sift_hiddenSizeL1 = 100;    % Layer 1 Hidden Size
sift_hiddenSizeL2 = 20;    % Layer 2 Hidden Size

surf_hiddenSizeL1 = 40;    % Layer 1 Hidden Size
surf_hiddenSizeL2 = 20;    % Layer 2 Hidden Size

sparsityParam = 0;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-2;         % weight decay parameter       
beta = 2;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load animal database files
load 'NUS/new_train_CH.mat'
load 'NUS/new_train_CM55.mat'  
load 'NUS/new_train_EDH.mat'  
load 'NUS/new_train_CORR.mat'  
load 'NUS/new_train_WT.mat'  
load 'NUS/NWO_Train_Labels.mat'


load 'NUS/new_test_CH.mat'         
load 'NUS/new_test_CM55.mat' 
load 'NUS/new_test_EDH.mat' 
load 'NUS/new_test_CORR.mat' 
load 'NUS/new_test_WT.mat' 
load 'NUS/NWO_TestLabel.mat' 

cq_trainData = new_train_CM55';
trainLabels = NWO_Train_Labels';
lss_trainData = new_train_CH';
phog_trainData = new_train_CORR';
sift_trainData = new_train_WT';
surf_trainData = new_train_EDH';


MaxIter = 200;
%%======================================================================
%% STEP 2: Train the first sparse autoencoder of cq_AE
%  Randomly initialize the parameters
cq_sae1Theta = initializeParameters(cq_hiddenSizeL1, cq_inputSize);
cq_sae1OptTheta = cq_sae1Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

fprintf('Train the first sparse autoencoder of cq_AE\n');
[cq_sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   cq_inputSize, cq_hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, cq_trainData), ...
                              cq_sae1Theta, options);
%%======================================================================
% STEP 2: Train the second sparse autoencoder of cq_AE
fprintf('Train the second sparse autoencoder of cq_AE\n');
[cq_sae1Features] = feedForwardAutoencoder(cq_sae1OptTheta, cq_hiddenSizeL1, ...
                                        cq_inputSize, cq_trainData);
%  Randomly initialize the parameters
cq_sae2Theta = initializeParameters(cq_hiddenSizeL2, cq_hiddenSizeL1);
cq_sae2OptTheta = cq_sae2Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[cq_sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   cq_hiddenSizeL1, cq_hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, cq_sae1Features), ...
                              cq_sae2Theta, options);
%%======================================================================
%%
%% STEP 2: Train the first sparse autoencoder of lss_AE
%  Randomly initialize the parameters
lss_sae1Theta = initializeParameters(lss_hiddenSizeL1, lss_inputSize);
lss_sae1OptTheta = lss_sae1Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

fprintf('Train the first sparse autoencoder of lss_AE\n');
[lss_sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   lss_inputSize, lss_hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, lss_trainData), ...
                              lss_sae1Theta, options);
%%======================================================================
% STEP 2: Train the second sparse autoencoder of lss_AE
fprintf('Train the second sparse autoencoder of lss_AE\n');
[lss_sae1Features] = feedForwardAutoencoder(lss_sae1OptTheta, lss_hiddenSizeL1, ...
                                        lss_inputSize, lss_trainData);
%  Randomly initialize the parameters
lss_sae2Theta = initializeParameters(lss_hiddenSizeL2, lss_hiddenSizeL1);
lss_sae2OptTheta = lss_sae2Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[lss_sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   lss_hiddenSizeL1, lss_hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, lss_sae1Features), ...
                              lss_sae2Theta, options);
%%======================================================================
%%
%% STEP 2: Train the first sparse autoencoder of phog_AE
%  Randomly initialize the parameters
phog_sae1Theta = initializeParameters(phog_hiddenSizeL1, phog_inputSize);
phog_sae1OptTheta = phog_sae1Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

fprintf('Train the first sparse autoencoder of phog_AE\n');
[phog_sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   phog_inputSize, phog_hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, phog_trainData), ...
                              phog_sae1Theta, options);
%%======================================================================
% STEP 2: Train the second sparse autoencoder of phog_AE
fprintf('Train the second sparse autoencoder of phog_AE\n');
[phog_sae1Features] = feedForwardAutoencoder(phog_sae1OptTheta, phog_hiddenSizeL1, ...
                                        phog_inputSize, phog_trainData);
%  Randomly initialize the parameters
phog_sae2Theta = initializeParameters(phog_hiddenSizeL2, phog_hiddenSizeL1);
phog_sae2OptTheta = phog_sae2Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[phog_sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   phog_hiddenSizeL1, phog_hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, phog_sae1Features), ...
                              phog_sae2Theta, options);
%%======================================================================
%%
%% STEP 2: Train the first sparse autoencoder of sift_AE
%  Randomly initialize the parameters
sift_sae1Theta = initializeParameters(sift_hiddenSizeL1, sift_inputSize);
sift_sae1OptTheta = sift_sae1Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

fprintf('Train the first sparse autoencoder of sift_AE\n');
[sift_sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   sift_inputSize, sift_hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, sift_trainData), ...
                              sift_sae1Theta, options);
%%======================================================================
% STEP 2: Train the second sparse autoencoder of sift_AE
fprintf('Train the second sparse autoencoder of sift_AE\n');
[sift_sae1Features] = feedForwardAutoencoder(sift_sae1OptTheta, sift_hiddenSizeL1, ...
                                        sift_inputSize, sift_trainData);
%  Randomly initialize the parameters
sift_sae2Theta = initializeParameters(sift_hiddenSizeL2, sift_hiddenSizeL1);
sift_sae2OptTheta = sift_sae2Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[sift_sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   sift_hiddenSizeL1, sift_hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sift_sae1Features), ...
                              sift_sae2Theta, options);
%%======================================================================
%%
%% STEP 2: Train the first sparse autoencoder of surf_AE
%  Randomly initialize the parameters
surf_sae1Theta = initializeParameters(surf_hiddenSizeL1, surf_inputSize);
surf_sae1OptTheta = surf_sae1Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

fprintf('Train the first sparse autoencoder of surf_AE\n');
[surf_sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   surf_inputSize, surf_hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, surf_trainData), ...
                              surf_sae1Theta, options);
%%======================================================================
% STEP 2: Train the second sparse autoencoder of surf_AE
fprintf('Train the second sparse autoencoder of surf_AE\n');
[surf_sae1Features] = feedForwardAutoencoder(surf_sae1OptTheta, surf_hiddenSizeL1, ...
                                        surf_inputSize, surf_trainData);
%  Randomly initialize the parameters
surf_sae2Theta = initializeParameters(surf_hiddenSizeL2, surf_hiddenSizeL1);
surf_sae2OptTheta = surf_sae2Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[surf_sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   surf_hiddenSizeL1, surf_hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, surf_sae1Features), ...
                              surf_sae2Theta, options);
%%======================================================================


%% STEP 3: Train the softmax classifier
[cq_sae2Features] = feedForwardAutoencoder(cq_sae2OptTheta, cq_hiddenSizeL2, ...
                                        cq_hiddenSizeL1, cq_sae1Features);
[lss_sae2Features] = feedForwardAutoencoder(lss_sae2OptTheta, lss_hiddenSizeL2, ...
                                        lss_hiddenSizeL1, lss_sae1Features);
[phog_sae2Features] = feedForwardAutoencoder(phog_sae2OptTheta, phog_hiddenSizeL2, ...
                                        phog_hiddenSizeL1, phog_sae1Features);                                    
[sift_sae2Features] = feedForwardAutoencoder(sift_sae2OptTheta, sift_hiddenSizeL2, ...
                                        sift_hiddenSizeL1,sift_sae1Features);                                    
[surf_sae2Features] = feedForwardAutoencoder(surf_sae2OptTheta, surf_hiddenSizeL2, ...
                                        surf_hiddenSizeL1, surf_sae1Features);                                                         
                                    
%%  �����Ƕ����Softmax��ѵ��                                    
%  Randomly initialize the parameters
fusionSize = cq_hiddenSizeL2 + lss_hiddenSizeL2 + phog_hiddenSizeL2 + sift_hiddenSizeL2 + surf_hiddenSizeL2;
fusionFeatures = [ cq_sae2Features' lss_sae2Features' phog_sae2Features' sift_sae2Features' surf_sae2Features' ]';
saeSoftmaxTheta = 0.005 * randn(fusionSize * numClasses, 1);

fprintf('Train the softmax\n');
lambda = 1e-4;
options.maxIter = MaxIter;
softmaxModel = softmaxTrain(fusionSize, numClasses, lambda, ...
                            fusionFeatures, trainLabels, options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);
% -------------------------------------------------------------------------
%% ����ںϲ��Ȩ�ز������ڷ��򴫲��㷨����
cq_saeSoftmaxOptTheta = saeSoftmaxOptTheta(1:cq_hiddenSizeL2*numClasses);

sizeIndex = cq_hiddenSizeL2*numClasses;
lss_saeSoftmaxOptTheta = saeSoftmaxOptTheta(sizeIndex+1:sizeIndex+lss_hiddenSizeL2*numClasses);

sizeIndex = sizeIndex + lss_hiddenSizeL2*numClasses;
phog_saeSoftmaxOptTheta = saeSoftmaxOptTheta(sizeIndex+1:sizeIndex+phog_hiddenSizeL2*numClasses);

sizeIndex = sizeIndex + phog_hiddenSizeL2*numClasses;
sift_saeSoftmaxOptTheta = saeSoftmaxOptTheta(sizeIndex+1:sizeIndex+sift_hiddenSizeL2*numClasses);

sizeIndex = sizeIndex + sift_hiddenSizeL2*numClasses;
surf_saeSoftmaxOptTheta = saeSoftmaxOptTheta(sizeIndex+1:sizeIndex+surf_hiddenSizeL2*numClasses);

%%======================================================================
%% STEP 5: Finetune softmax model
%-- cq_stack
cq_stack = cell(2,1);
cq_stack{1}.w = reshape(cq_sae1OptTheta(1:cq_hiddenSizeL1*cq_inputSize), ...
                     cq_hiddenSizeL1, cq_inputSize);
cq_stack{1}.b = cq_sae1OptTheta(2*cq_hiddenSizeL1*cq_inputSize+1:2*cq_hiddenSizeL1*cq_inputSize+cq_hiddenSizeL1);
cq_stack{2}.w = reshape(cq_sae2OptTheta(1:cq_hiddenSizeL2*cq_hiddenSizeL1), ...
                     cq_hiddenSizeL2, cq_hiddenSizeL1);
cq_stack{2}.b = cq_sae2OptTheta(2*cq_hiddenSizeL2*cq_hiddenSizeL1+1:2*cq_hiddenSizeL2*cq_hiddenSizeL1+cq_hiddenSizeL2);
% Initialize the parameters for the deep model
[cq_stackparams, cq_netconfig] = stack2params(cq_stack);
cq_stackedAETheta = [ cq_saeSoftmaxOptTheta ; cq_stackparams ];
%% ---------------------- YOUR CODE HERE  ---------------------------------
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
lambda = 3e-3;   %
fprintf('Fine-tuning cq_stacked autoencoder\n');
[cq_stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                   cq_hiddenSizeL1, cq_hiddenSizeL2, ...
                                   numClasses,cq_netconfig,lambda, ...
                                   cq_trainData,trainLabels), ...
                              cq_stackedAETheta, options);
% -------------------------------------------------------------------------
%-- lss_stack
lss_stack = cell(2,1);
lss_stack{1}.w = reshape(lss_sae1OptTheta(1:lss_hiddenSizeL1*lss_inputSize), ...
                     lss_hiddenSizeL1, lss_inputSize);
lss_stack{1}.b = lss_sae1OptTheta(2*lss_hiddenSizeL1*lss_inputSize+1:2*lss_hiddenSizeL1*lss_inputSize+lss_hiddenSizeL1);
lss_stack{2}.w = reshape(lss_sae2OptTheta(1:lss_hiddenSizeL2*lss_hiddenSizeL1), ...
                     lss_hiddenSizeL2, lss_hiddenSizeL1);
lss_stack{2}.b = lss_sae2OptTheta(2*lss_hiddenSizeL2*lss_hiddenSizeL1+1:2*lss_hiddenSizeL2*lss_hiddenSizeL1+lss_hiddenSizeL2);
% Initialize the parameters for the deep model
[lss_stackparams, lss_netconfig] = stack2params(lss_stack);
lss_stackedAETheta = [ lss_saeSoftmaxOptTheta ; lss_stackparams ];
%% ---------------------- YOUR CODE HERE  ---------------------------------
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
lambda = 3e-3;   %
fprintf('Fine-tuning lss_stacked autoencoder\n');
[lss_stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                   lss_hiddenSizeL1, lss_hiddenSizeL2, ...
                                   numClasses,lss_netconfig,lambda, ...
                                   lss_trainData,trainLabels), ...
                              lss_stackedAETheta, options);
% -------------------------------------------------------------------------
%-- phog_stack
phog_stack = cell(2,1);
phog_stack{1}.w = reshape(phog_sae1OptTheta(1:phog_hiddenSizeL1*phog_inputSize), ...
                     phog_hiddenSizeL1, phog_inputSize);
phog_stack{1}.b = phog_sae1OptTheta(2*phog_hiddenSizeL1*phog_inputSize+1:2*phog_hiddenSizeL1*phog_inputSize+phog_hiddenSizeL1);
phog_stack{2}.w = reshape(phog_sae2OptTheta(1:phog_hiddenSizeL2*phog_hiddenSizeL1), ...
                     phog_hiddenSizeL2, phog_hiddenSizeL1);
phog_stack{2}.b = phog_sae2OptTheta(2*phog_hiddenSizeL2*phog_hiddenSizeL1+1:2*phog_hiddenSizeL2*phog_hiddenSizeL1+phog_hiddenSizeL2);
% Initialize the parameters for the deep model
[phog_stackparams, phog_netconfig] = stack2params(phog_stack);
phog_stackedAETheta = [ phog_saeSoftmaxOptTheta ; phog_stackparams ];
%% ---------------------- YOUR CODE HERE  ---------------------------------
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
lambda = 3e-3;   %
fprintf('Fine-tuning phog_stacked autoencoder\n');
[phog_stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                   phog_hiddenSizeL1, phog_hiddenSizeL2, ...
                                   numClasses,phog_netconfig,lambda, ...
                                   phog_trainData,trainLabels), ...
                              phog_stackedAETheta, options);
% -------------------------------------------------------------------------
%-- sift_stack
sift_stack = cell(2,1);
sift_stack{1}.w = reshape(sift_sae1OptTheta(1:sift_hiddenSizeL1*sift_inputSize), ...
                     sift_hiddenSizeL1, sift_inputSize);
sift_stack{1}.b = sift_sae1OptTheta(2*sift_hiddenSizeL1*sift_inputSize+1:2*sift_hiddenSizeL1*sift_inputSize+sift_hiddenSizeL1);
sift_stack{2}.w = reshape(sift_sae2OptTheta(1:sift_hiddenSizeL2*sift_hiddenSizeL1), ...
                     sift_hiddenSizeL2, sift_hiddenSizeL1);
sift_stack{2}.b = sift_sae2OptTheta(2*sift_hiddenSizeL2*sift_hiddenSizeL1+1:2*sift_hiddenSizeL2*sift_hiddenSizeL1+sift_hiddenSizeL2);
% Initialize the parameters for the deep model
[sift_stackparams, sift_netconfig] = stack2params(sift_stack);
sift_stackedAETheta = [ sift_saeSoftmaxOptTheta ; sift_stackparams ];
%% ---------------------- YOUR CODE HERE  ---------------------------------
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
lambda = 3e-3;   %
fprintf('Fine-tuning sift_stacked autoencoder\n');
[sift_stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                   sift_hiddenSizeL1, sift_hiddenSizeL2, ...
                                   numClasses,sift_netconfig,lambda, ...
                                   sift_trainData,trainLabels), ...
                              sift_stackedAETheta, options);
% -------------------------------------------------------------------------
%-- surf_stack
surf_stack = cell(2,1);
surf_stack{1}.w = reshape(surf_sae1OptTheta(1:surf_hiddenSizeL1*surf_inputSize), ...
                     surf_hiddenSizeL1, surf_inputSize);
surf_stack{1}.b = surf_sae1OptTheta(2*surf_hiddenSizeL1*surf_inputSize+1:2*surf_hiddenSizeL1*surf_inputSize+surf_hiddenSizeL1);
surf_stack{2}.w = reshape(surf_sae2OptTheta(1:surf_hiddenSizeL2*surf_hiddenSizeL1), ...
                     surf_hiddenSizeL2, surf_hiddenSizeL1);
surf_stack{2}.b = surf_sae2OptTheta(2*surf_hiddenSizeL2*surf_hiddenSizeL1+1:2*surf_hiddenSizeL2*surf_hiddenSizeL1+surf_hiddenSizeL2);
% Initialize the parameters for the deep model
[surf_stackparams, surf_netconfig] = stack2params(surf_stack);
surf_stackedAETheta = [ surf_saeSoftmaxOptTheta ; surf_stackparams ];
%% ---------------------- YOUR CODE HERE  ---------------------------------
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = MaxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
lambda = 3e-3;   %
fprintf('Fine-tuning surf_stacked autoencoder\n');
[surf_stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                   surf_hiddenSizeL1, surf_hiddenSizeL2, ...
                                   numClasses,surf_netconfig,lambda, ...
                                   surf_trainData,trainLabels), ...
                              surf_stackedAETheta, options);
% -------------------------------------------------------------------------

%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set

testLabels = BB';
cq_testData = new_test_CM55';
lss_testData = new_test_CH';
phog_testData = new_test_CORR';
sift_testData = new_test_WT';
surf_testData = new_test_EDH';

[maxy ylabel]= max(testLabels);

%testLabels(testLabels == 0) = 10; % Remap 0 to 10

netconfig = {cq_netconfig, lss_netconfig,phog_netconfig,sift_netconfig,surf_netconfig};
stackedAETheta = {cq_stackparams,lss_stackparams,phog_stackparams,sift_stackparams,surf_stackparams};
inputSize = [cq_inputSize; lss_inputSize;phog_inputSize;sift_inputSize;surf_inputSize];
hiddenSizeL2 = [cq_hiddenSizeL2; lss_hiddenSizeL2;phog_hiddenSizeL2;sift_hiddenSizeL2;surf_hiddenSizeL2];
testData = { cq_testData, lss_testData, phog_testData, sift_testData, surf_testData};

[pred] = NUS_multistackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData,saeSoftmaxOptTheta);
                      

acc = mean(ylabel(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% �������ǽ�Ȩ�ؾ���ֿ�?
% cq type
cq_softmaxTheta = reshape(cq_stackedAEOptTheta(1:cq_hiddenSizeL2*numClasses), numClasses, cq_hiddenSizeL2);
cq_stackparams  = cq_stackedAEOptTheta(cq_hiddenSizeL2*numClasses+1:end);
% lss type
lss_softmaxTheta = reshape(lss_stackedAEOptTheta(1:lss_hiddenSizeL2*numClasses), numClasses, lss_hiddenSizeL2);
lss_stackparams  = lss_stackedAEOptTheta(lss_hiddenSizeL2*numClasses+1:end);
% phog type
phog_softmaxTheta = reshape(phog_stackedAEOptTheta(1:phog_hiddenSizeL2*numClasses), numClasses, phog_hiddenSizeL2);
phog_stackparams  = phog_stackedAEOptTheta(phog_hiddenSizeL2*numClasses+1:end);
% sift type
sift_softmaxTheta = reshape(sift_stackedAEOptTheta(1:sift_hiddenSizeL2*numClasses), numClasses, sift_hiddenSizeL2);
sift_stackparams  = sift_stackedAEOptTheta(sift_hiddenSizeL2*numClasses+1:end);
% surf type
surf_softmaxTheta = reshape(surf_stackedAEOptTheta(1:surf_hiddenSizeL2*numClasses), numClasses, surf_hiddenSizeL2);
surf_stackparams  = surf_stackedAEOptTheta(surf_hiddenSizeL2*numClasses+1:end);

% �ϳ�fusion
stackedAETheta = {cq_stackparams,lss_stackparams,phog_stackparams,sift_stackparams,surf_stackparams};
SoftmaxOptTheta = [cq_softmaxTheta lss_softmaxTheta phog_softmaxTheta sift_softmaxTheta surf_softmaxTheta ];
stackedAEOptTheta = {cq_stackedAEOptTheta,lss_stackedAEOptTheta,phog_stackedAEOptTheta,sift_stackedAEOptTheta,surf_stackedAEOptTheta };

[pred] = NUS_multistackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData,SoftmaxOptTheta);
                     

acc = mean(ylabel(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

