function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
hypothesis=X*theta;
aH=sigmoid(hypothesis);
jArr=((-y).*(log(aH)))-((1-y).*(log(1-aH)));
J=((sum(jArr))/m)+(lambda/(2*m))*(sum(theta(2:n,:).^2));
err=aH-y;
k = X'*err;
k=(k/m);
k2=(theta*(lambda/m));
k2(1)=0;
grad=k+k2;



% =============================================================

grad = grad(:);

end
