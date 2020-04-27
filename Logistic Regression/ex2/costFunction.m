function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta); 
J = 0;
grad = zeros(size(theta));

hypothesis=X*theta;
aH=sigmoid(hypothesis);
jArr=((-y).*(log(aH)))-((1-y).*(log(1-aH)));
J=((sum(jArr))/m);
for i =1:n
  t=aH-y;
  k=t.*X(:,i);
  grad(i)=(sum(k)/m);
endfor









% =============================================================

end
