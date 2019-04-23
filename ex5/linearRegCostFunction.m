function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
m = length(y); % number of training examples
J = 1/(2.*m) * sum((sum(X .* theta', 2)-y).^2) + sum(lambda/2/m.*theta.^2);
J = J - lambda/2/m * theta(1)^2;

grad = zeros(size(theta));
htheta = sum(theta'.* X,2);
grad = 1/m * sum((htheta-y) .* X, 1) .+ (lambda/m) * theta';
grad(1) = grad(1) - lambda/m * theta(1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
