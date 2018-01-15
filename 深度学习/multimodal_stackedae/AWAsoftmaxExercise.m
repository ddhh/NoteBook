clear all;
numClasses = 10;
cq_inputSize = 76; % Size of input vector cq 2688*18305
lss_inputSize = 216;
phog_inputSize = 64;
sift_inputSize = 240;
surf_inputSize = 47;
rgsift_inputSize = 6;
lambda = 1e-4; % Weight decay parameter

%%======================================================================
%% STEP 1: Load data
load ('D:\ML Matlab\Data\HW\mfeat\mfeat-fou');
load ('D:\ML Matlab\Data\HW\mfeat\mfeat-fac');
load ('D:\ML Matlab\Data\HW\mfeat\mfeat-kar');
load ('D:\ML Matlab\Data\HW\mfeat\mfeat-pix');
load ('D:\ML Matlab\Data\HW\mfeat\mfeat-zer');
load ('D:\ML Matlab\Data\HW\mfeat\mfeat-mor');

label = [10*ones(1,200) ones(1,200) 2*ones(1,200) 3*ones(1,200) 4*ones(1,200) 5*ones(1,200) 6*ones(1,200) 7*ones(1,200) 8*ones(1,200) 9*ones(1,200)]';

cq_try=label([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960]);%1600
trainLabels = full(sparse(cq_try, 1:1600, 1));
  
cq_trx = mfeat_fou([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960],:)';
lss_trx = mfeat_fac([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960],:)';
phog_trx = mfeat_kar([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960],:)';
sift_trx = mfeat_pix([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960],:)';
surf_trx = mfeat_zer([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960],:)';
rgsift_trx = mfeat_mor([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960],:)';

[cq_trainData hwave hwsigma] = featureNormalize(cq_trx);
[lss_trainData hwave hwsigma] = featureNormalize(lss_trx);
[phog_trainData hwave hwsigma] = featureNormalize(phog_trx);
[sift_trainData hwave hwsigma] = featureNormalize(sift_trx);
[surf_trainData hwave hwsigma] = featureNormalize(surf_trx);
[rgsift_trainData hwave hwsigma] = featureNormalize(rgsift_trx);

inputSize = cq_inputSize + lss_inputSize + phog_inputSize + sift_inputSize + surf_inputSize + rgsift_inputSize;
inputData = [cq_trainData;lss_trainData;phog_trainData;sift_trainData;surf_trainData;rgsift_trainData];
labels = trainLabels;
%labels(labels==0) = 10; % Remap 0 to 10


% For debugging purposes, you may wish to reduce the size of the input data
% in order to speed up gradient checking. 
% Here, we create synthetic dataset using random data for testing

DEBUG = false; % Set DEBUG to true when debugging.
if DEBUG
    inputSize = 8;
    inputData = randn(8, 100);
    labels = randi(10, 100, 1);
end

% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1);

%%======================================================================
%% STEP 2: Implement softmaxCost
%
%  Implement softmaxCost in softmaxCost.m. 

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
                                     
%%======================================================================
%% STEP 3: Gradient checking
%
%  As with any learning algorithm, you should always check that your
%  gradients are correct before learning the parameters.
% 

if DEBUG
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                    inputSize, lambda, inputData, labels), theta);

    % Use this to visually compare the gradients side by side
    disp([numGrad grad]); 

    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    % The difference should be small. 
    % In our implementation, these values are usually less than 1e-7.

    % When your gradients are correct, congratulations!
end

%%======================================================================
%% STEP 4: Learning parameters
%
%  Once you have verified that your gradients are correct, 
%  you can start training your softmax regression code using softmaxTrain
%  (which uses minFunc).

options.maxIter = 600;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);
                          
% Although we only use 100 iterations here to train a classifier for the 
% MNIST data set, in practice, training for more iterations is usually
% beneficial.

%%======================================================================
%% STEP 5: Testing
cq_ty=label([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000]);
testLabels = full(sparse(cq_ty, 1:400, 1));
    
cq_tx = mfeat_fou([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000],:)';
lss_tx = mfeat_fac([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000],:)';
phog_tx = mfeat_kar([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000],:)';
sift_tx = mfeat_pix([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000],:)';
surf_tx = mfeat_zer([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000],:)';
rgsift_tx = mfeat_mor([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000],:)';

[cq_testData hwave hwsigma] = featureNormalize(cq_tx);
[lss_testData hwave hwsigma] = featureNormalize(lss_tx);
[phog_testData hwave hwsigma] = featureNormalize(phog_tx);
[sift_testData hwave hwsigma] = featureNormalize(sift_tx);
[surf_testData hwave hwsigma] = featureNormalize(surf_tx);
[rgsift_testData hwave hwsigma] = featureNormalize(rgsift_tx);

[maxy ylabel]= max(testLabels);

images = [cq_testData;lss_testData;phog_testData;sift_testData;surf_testData;rgsift_testData];
labels = testLabels;
inputData = images;


% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);
[maxy ylabel]= max(labels);
acc = mean(ylabel(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);


