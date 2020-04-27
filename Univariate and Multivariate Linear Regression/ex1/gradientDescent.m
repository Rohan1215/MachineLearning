function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
   
    a=X*theta;
    b=a-y;
    finalZero=(sum(b)*(alpha)/m);
    littleX=X(:,2);
    c=b.*littleX;
    finalOne=(sum(c)*(alpha)/m);
    tempZero=theta(1)-finalZero;
    tempOne=theta(2)-finalOne;
    theta(1)=tempZero;
    theta(2)=tempOne;

    

    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
