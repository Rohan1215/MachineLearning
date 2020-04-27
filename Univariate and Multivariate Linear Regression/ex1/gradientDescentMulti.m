function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n=size(X,2);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    
    hTheta=X*theta;
    totalError=hTheta-y;
    for feature=1:n
      temp=(totalError.*(X(:,feature)));
      temp2=sum(temp);
      temp2=temp2*(alpha)/(m);
      theta(feature)=theta(feature)-temp2;
    endfor
    if iter<150
    disp(computeCost(X,y,theta));
    end










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
