function [Compoenents] = F_Transform(X, n)

% input: 
%       X: input vector
%       n: output vector length n; 
% output:
%       Compoenents: transform vector

Compoenents = zeros(n,1);
len = size(X,1);
a = 1;
b = len;
for k = 1:n
    
    temp1 = 0;
    temp2 = 0;
    for j=1:len
        temp1 = temp1+ X(j)*Akfunc(j,k,a,b,n);
        temp2 = temp2 + Akfunc(j,k,a,b,n);
    end
    Compoenents(k) = temp1/temp2;
end

end


