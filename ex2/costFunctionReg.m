% -*- octave -*-
function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.
m = length(y); % number of training examples
htheta = sigmoid(sum(theta'.*X, 2));
J = 1/m * sum(-y.*log(htheta) .- (1 - y) .* log(1.0 - htheta)) + sum(lambda/2/m.*theta.^2);
J = J - lambda/2/m * theta(1)^2;
grad = 1/m * sum((htheta-y) .* X, 1) .+ (lambda/m) * theta';
grad(1) = grad(1) - lambda/m * theta(1);
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

end
