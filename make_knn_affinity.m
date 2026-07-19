function A = make_knn_affinity(X, k, opts)
    if nargin < 3
        opts = struct();
    end
    if ~isfield(opts, 'block_size'), opts.block_size = 1000; end

    [n, ~] = size(X);
    if k < 1 || k >= n
        error('k must satisfy 1 <= k < number of samples.');
    end

    row_norm = sqrt(sum(X.^2, 2));
    row_norm(row_norm == 0) = 1;
    X = X ./ row_norm;

    if exist('knnsearch', 'file') == 2
        idx = knnsearch(X, X, 'K', k + 1);
        idx = idx(:, 2:end);
    else
        idx = zeros(n, k);
        xnorm2 = sum(X.^2, 2)';
        for first = 1:opts.block_size:n
            last = min(n, first + opts.block_size - 1);
            Xb = X(first:last, :);
            bnorm2 = sum(Xb.^2, 2);
            D = bnorm2 + xnorm2 - 2 * (Xb * X');
            D = max(D, 0);

            rows = (first:last)';
            local_rows = (1:numel(rows))';
            D(sub2ind(size(D), local_rows, rows)) = inf;

            [~, block_idx] = mink(D, k, 2);
            idx(first:last, :) = block_idx;
        end
    end

    row_id = repelem((1:n)', k, 1);
    col_id = reshape(idx', [], 1);
    A0 = sparse(row_id, col_id, 1, n, n);

    A = spones(max(A0, A0'));
    A = A - spdiags(diag(A), 0, n, n);
end
