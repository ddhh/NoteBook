
%% STEP 0: Here we provide the relevant parameters values that will
clear all;clc

inputSize = 303; % Size of input vector (MNIST images are 28x28)
numClasses = 3883;     % Number of classes (MNIST images fall into 10 classes)

hiddenSizeL1 = 100;    % Layer 1 Hidden Size
hiddenSizeL2 = 100;    % Layer 2 Hidden Size
hiddenSizeL3 = 300;
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-2;         % weight decay parameter       
beta = 1;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
load 'D:\ML Matlab\ZHENGXIN\user_train.mat' 
item = size(user_train,1);

tri = randperm(item,20000);
tei = randperm(item,5000);

images = user_train(tri,1:end-1)';
%images = [m1;m2;m3;m4;m5];
labels = full(sparse(user_train(tri,end), 1:20000, 1));
size(labels)


trainLabels = labels;

[phog_trainData mu sigma] = featureNormalize(images);

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);
sae1OptTheta = sae1Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
fprintf('Train the first phog_sparse autoencoder\n');
lambda = 1e-1;
[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, phog_trainData), ...
                              sae1Theta, options);
% -------------------------------------------------------------------------

%%======================================================================
%% STEP 2: Train the second sparse autoencoder
fprintf('Train the second sparse autoencoder\n');
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, phog_trainData);
%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

sae2OptTheta = sae2Theta;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
lambda = 2e-2; 
[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                              sae2Theta, options);
% -------------------------------------------------------------------------

%%======================================================================
%% STEP 3: Train the softmax classifier
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);

fprintf('Train the softmax\n');
lambda = 1e-3;

options.maxIter = 100;
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
lambda = 2e-2;   %


fprintf('Fine-tuning stacked autoencoder\n');
[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   numClasses,netconfig,lambda, ...
                                   phog_trainData,trainLabels), ...
                              stackedAETheta, options);
% -------------------------------------------------------------------------
%%======================================================================
%% STEP 6: Test 



images = user_train(tei,1:end-1)';
%images = [m1;m2;m3;m4;m5];
labels = full(sparse(user_train(tei,end), 1:5000, 1));
size(labels)



testLabels = labels;
[maxy ylabel]= max(testLabels);

[testData mu sigma] = featureNormalize(images);

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(ylabel(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(ylabel(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

