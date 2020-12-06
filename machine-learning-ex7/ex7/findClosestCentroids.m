function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

[m n] = size(X);

for i=1:m
  x = X(i,:);
  distances = zeros(K);
  for j=1:K
    c = centroids(j,:);
    % Manhattan Distance between the centroid and example x
    for col=1:n
      distances(j) += (c(col) - x(col)) ^ 2;
    end
  end
  [_, idx_smallest] = min(distances);
  idx(i) = idx_smallest(1); % for some reason idx_smallest is an array
end


% =============================================================

end

