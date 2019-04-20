function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_one_hot = zeros(size(y, 1), num_labels);
for index = 1:size(y, 1)
  y_one_hot(index, y(index)) = 1;
end

sum_m = 0;
for index = 1:m # For each training case
  sum_k = 0;

  # Feed Forward
  a1 = transpose(X(index,:));
  a1 = [1;a1]; % bias unit
  a2 = sigmoid(Theta1 * a1);
  a2 = [1;a2]; % bias unit
  a3 = sigmoid(Theta2 * a2);

  # Cost for this case
  for k = 1:num_labels
    y_ = y_one_hot(index, k);
    h0 = a3(k);
    sum_k += -y_ * log(h0) - (1 - y_) * log(1 - h0);
  end
  sum_m += sum_k;
end

Thetha1_nobias = Theta1(1:hidden_layer_size, 2:(input_layer_size + 1)); % 25x400
Thetha2_nobias = Theta2(1:num_labels, 2:(hidden_layer_size + 1)); % 10x25
regularization = lambda / (2 * m) * ( ...
  sum(sum(Thetha1_nobias .^ 2)) + ...
  sum(sum(Thetha2_nobias .^ 2)) ...
);

J = 1 / m * sum_m + regularization;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
