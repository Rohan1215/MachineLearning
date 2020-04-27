function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);

J = 0;
grad = zeros(size(theta));

hypothesis=X*theta;
aH=sigmoid(hypothesis);
jArr=((-y).*(log(aH)))-((1-y).*(log(1-aH)));
J=((sum(jArr))/m)+(lambda/(2*m))*(sum(theta(2:n,:).^2));
for j =1:n
  t=aH-y;
  k=t.*X(:,j);
  if j>1
    k=k+(lambda/m)*(theta(j));
  endif
  grad(j)=(sum(k)/m);
endfor




% =============================================================

end
