function X = normalize_rows(X)
%NORMALIZE_ROWS Row-wise L2 normalization with zero-row protection.
    scale = sqrt(sum(X.^2, 2));
    scale(scale < eps) = 1;
    X = X ./ scale;
end
