function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_data = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_data = [0.01 0.03 0.1 0.3 1 3 10 30]';

scores = zeros(size(C_data),size(sigma_data));

score_min = 1e100;
Cmin = 0;
smin = 0;

for i = 1:size(C_data,1)
  C_trial = C_data(i);
  for j = 1:size(sigma_data,1)
    sigma_trial = sigma_data(j);
    model = svmTrain(X, y, C_trial, @(x1, x2) gaussianKernel(x1, x2, sigma_trial));
    predictions = svmPredict(model, Xval);
    scores(i,j) = mean(double(predictions ~= yval));
    if scores(i,j) < score_min 
      score_min = scores(i,j);
      Cmin = C_trial;
      smin = sigma_trial;
    endif
    [i j scores(i,j)]
  endfor
endfor


C = Cmin;
sigma = smin;

% =========================================================================
C
sigma
min(scores(:))
end
