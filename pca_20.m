function score_20 = pca_20(data)
%PCA_20 Summary of this function goes here
% Perform PCA
[coeff, score, ~, ~, explained] = pca(data);

% Get the first 20 principal components
coeff_20 = coeff(:, 1:20); % Principal component coefficients
score_20 = score(:, 1:20); % Projected data on the first 20 PCs
explained_20 = explained(1:20); % Variance explained by the first 20 PCs

% Display variance explained by the first 20 PCs
% disp('Variance explained by the first 20 PCs:');
% disp(explained_20);
end

