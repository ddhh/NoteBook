%% CS294A/CS294W Softmax Exercise 


%clear all;
inputSize = 303; % Size of input vector (MNIST images are 28x28)
numClasses = 3883;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-2; % Weight decay parameter

%%======================================================================
%% STEP 1: Load data

%load 'D:\ML Matlab\ZHENGXIN\usrdata.mat'  
%load 'D:\ML Matlab\ZHENGXIN\usrLabel.mat'  


%[m1 mu sigma] = featureNormalize(usrdata');

tri = randperm(100000,10000);
%tei = randperm(100000,10000);

images = user_train(tri,1:end-1)';
%images = [m1;m2;m3;m4;m5];
labels = full(sparse(user_train(tri,end), 1:10000, 1));
size(labels)




inputData = images;

%[inputData mu sigma] = featureNormalize(images);

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
options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);
                          
% Although we only use 100 iterations here to train a classifier for the 
% MNIST data set, in practice, training for more iterations is usually
% beneficial.

%%======================================================================
%% STEP 5: Testing


%[inputData mu sigma] = featureNormalize(images);


% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);
[maxy ylabel]= max(labels);
acc = mean(ylabel(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);


