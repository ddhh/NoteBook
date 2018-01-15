function [value ] = Akfunc(x,k,a,b,n)

% x the input xi
% k the k-th function Ak(x)

h = (b-a)/(n-1);

kk = 1:n;
XK = a + h*(kk - 1);

if k==1
    if x >= XK(1) && x < XK(2)
        value = 1-(x-XK(1))/h;
    else
        value = 0;
    end
end

if k == n
    if x >= XK(n-1) && x <= XK(n)
        value = (x-XK(n-1))/h;
    else
        value = 0;
    end
end

if k>1 && k<n
    
   if x >= XK(k-1) && x < XK(k)
        value = (x-XK(k-1))/h;
   else if x >= XK(k) && x <XK(k+1)
           value = 1-(x-XK(k))/h;
       else
        value = 0;
       end
   end
end
end
        