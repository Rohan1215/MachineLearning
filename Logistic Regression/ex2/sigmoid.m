function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

a=-z;
eToZ=e.^a;
g=eToZ+1;
g=1./g;




% =============================================================

end
