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

[m1 mu sigma] = featureNormalize(new_train_CM55');
[m2 mu sigma] = featureNormalize(new_train_CH');
[m3 mu sigma] = featureNormalize(new_train_CORR');
[m4 mu sigma] = featureNormalize(new_train_WT');
[m5 mu sigma] = featureNormalize(new_train_EDH');

images = [ new_train_CH';new_train_CORR';new_train_WT';new_train_CM55'; new_train_EDH'];
%images = [m1;m2;m3;m4;m5];
labels = NWO_Train_Labels';
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
options.maxIter = 400;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);
                          
% Although we only use 100 iterations here to train a classifier for the 
% MNIST data set, in practice, training for more iterations is usually
% beneficial.

%%======================================================================
%% STEP 5: Testing

load 'D:\ML Matlab\Data\NUS-WIDE\new_test_CM55.mat'  
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_CH.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_CORR.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_WT.mat'    
load 'D:\ML Matlab\Data\NUS-WIDE\new_test_EDH.mat'  

load 'D:\ML Matlab\Data\NUS-WIDE\NWO_TestLabel.mat'          % ï¿½ï¿½ï¿½ï¿½Ñµï¿½ï¿½ï¿½ï¿½ï¿?load 'animal/cq_try.mat'

images = [ new_test_CH';new_test_CORR';new_test_WT'; new_test_CM55';new_test_EDH'];
labels = BB';
inputData = images;
%[inputData mu sigma] = featureNormalize(images);


% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);
[maxy ylabel]= max(labels);
acc = mean(ylabel(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);


