function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly

h = X * theta;
J = 1 / (2 * m) * (sum(sum((h - y) .^ 2)));
% Regularization
theta_ = theta;
theta_(1) = 0;
J += lambda / (2 * m) * (sum(sum(theta_ .^ 2)));

grad = 1 / m * transpose(X) * (h - y) + (lambda / m) * theta_;

% =========================================================================

grad = grad(:);

end
