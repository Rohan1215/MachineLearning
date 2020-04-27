function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
le=length(X);

     

for i = 1:size(X,2)
  disp(i);
  m=sum(X(:,i))/le;
  s=std(X(:,i));
 % disp(m)
 % disp(s);
 % disp("-----");
  mu(i)=m;
  sigma(i)=s;
  X_norm(:,i)=X(:,i)-m;
  X_norm(:,i)=X_norm(:,i)/s;
  %disp(sum(X_norm(:,i)));
  %disp(std(X_norm(:,i)));
endfor







% ============================================================

end
