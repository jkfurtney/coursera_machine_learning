%% -*- mode: octave -*-
function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X];
p = zeros(size(X, 1), 1);

step1 = sigmoid(X*Theta1');
step1 = [ones(size(step1,1),1) step1];
step2 = sigmoid(step1*Theta2');
[m, im] = max(step2,[],2);
p=im;

end
