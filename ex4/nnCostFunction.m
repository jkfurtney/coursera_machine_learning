%% -*- mode: octave -*-
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
Theta1_grad = zeros(size(Theta1));  %  25, 401 
Theta2_grad = zeros(size(Theta2));  %  10, 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X];
a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(a2*Theta2');

%newy2 = [y==1 y==2 y==3 y==4 y==5 y==6 y==7 y==8 y==9 y==10];
newy = zeros(size(a3));
for i = 1:m 
  newy(i,y(i))=1;  
end

J = 1/m * sum(-newy.*log(a3) .- (1 - newy) .* log(1.0 - a3));
J = sum(J);

reg_term = lambda/(2.0*m)*(sum(sum(Theta1(:,2:end).^2))+ sum(sum(Theta2(:,2:end).^2)));
J = J + reg_term;
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

D1 = zeros(hidden_layer_size, input_layer_size+1);
D2 = zeros(num_labels, hidden_layer_size+1);
% Theta1 is 25, 401 maps from layer 1 to layer 2, Theta2 is 10,26
for t = 1:m
  a1 = X(t,:);              % 1, 401   single row
  z2 = Theta1 * a1';        % 25, 1  single column
  a2 = [1; sigmoid(z2)];    % 26, 1
  z3 = Theta2 * a2;         % 10, 1 single column
  a3 = sigmoid(z3);         % 10, 1  network result
  delta_3 = a3 - y(t);      % 10, 1
  %delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]);  % 25, 1
  %delta_2 = delta_2(2:end);
  delta_2 = (Theta2' * delta_3)(2:end) .* sigmoidGradient(z2);  % 25, 1
  D2 = D2 + delta_3*a2';
  D1 = D1 + delta_2*a1;
end
Theta1_grad = (1/m)*D1;
Theta2_grad = (1/m)*D2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
