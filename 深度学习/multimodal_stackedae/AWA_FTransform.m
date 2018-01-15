clear all;

numClasses = 50;
cq_inputSize = 2688; % Size of input vector cq 2688

lambda = 1e-4; % Weight decay parameter

%%======================================================================
%% STEP 1: Load data
load 'D:\ML Matlab\Data\AWA\cq_trx.mat'          % ï¿½ï¿½ï¿½ï¿½Ñµï¿½ï¿½ï¿½ï¿½ï¿?
load 'D:\ML Matlab\Data\AWA\cq_try.mat'

%[cq_trainData hwave hwsigma] = featureNormalize(cq_trx);
n = 200;
m = size(cq_trx,2)
cq_trainData = zeros(n,m);

for i=1:m
    cq_trainData(:,i) = F_Transform(cq_trx(:,i),n);
    i
end
fprintf('F-Transform finished\n');

inputSize = n;
inputData = cq_trainData;
labels = trainLabels;

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
load 'D:\ML Matlab\Data\AWA\cq_tx.mat'           % ï¿½ï¿½ï¿½Ø²ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
load 'D:\ML Matlab\Data\AWA\cq_ty.mat'

testLabels = cq_ty;


m = size(cq_tx,2);
cq_testData = zeros(n,m);

for i=1:m
    cq_testData(:,i) = F_Transform(cq_tx(:,i),n);
end
fprintf('F-Transform finished\n');

%[cq_testData hwave hwsigma] = featureNormalize(cq_tx);

[maxy ylabel]= max(testLabels);

images = cq_testData;
labels = testLabels;
inputData = images;


% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);
[maxy ylabel]= max(labels);
acc = mean(ylabel(:) == pred(:));
fprintf('Accuracy: %0.4f%%\n', acc * 100);


