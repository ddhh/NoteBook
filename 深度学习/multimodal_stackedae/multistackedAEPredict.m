function [pred] = multistackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data, softmaxOTheta)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter
% We first extract the part which compute the softmax gradient

Size = sum(hiddenSize);
softmaxTheta = reshape(softmaxOTheta(1:Size*numClasses), numClasses, Size);

%%
fusionData = zeros(Size,size(data{1},2));
index = 1;
for i=1:6
    stack = params2stack(theta{i}(1:end), netconfig{i});
    nl = numel(stack);
    z = cell(nl+1,1);
    a = cell(nl+1, 1);
    a{1} = data{i};
    for d = 1:nl    
        z{d+1} = stack{d}.w*a{d} + repmat(stack{d}.b,1,size(a{d},2));
        a{d+1} = sigmoid(z{d+1});
    end

    
    one_featureData = a{nl+1}; 
    %if i== 1
        fusionData(index:index+hiddenSize(i)-1,:) = one_featureData;
    %end
   
    index = index + hiddenSize(i);
end
%%
resu = exp(softmaxTheta*fusionData);

[maxy ylabel]= max(resu);
pred = ylabel;




% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
