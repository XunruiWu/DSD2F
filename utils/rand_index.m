function ri = rand_index(labels_true, labels_pred)
%RAND_INDEX Ordinary Rand index in [0,1], computed from a contingency table.

    labels_true = labels_true(:);
    labels_pred = labels_pred(:);
    if numel(labels_true) ~= numel(labels_pred)
        error('RAND_INDEX:SizeMismatch', 'The two label vectors must have equal length.');
    end
    n = numel(labels_true);
    if n < 2
        ri = 1;
        return;
    end

    [~,~,a] = unique(labels_true, 'stable');
    [~,~,b] = unique(labels_pred, 'stable');
    contingency = accumarray([a,b], 1);

    choose2 = @(x) x .* (x - 1) / 2;
    tp = sum(choose2(contingency(:)));
    row_pairs = sum(choose2(sum(contingency,2)));
    col_pairs = sum(choose2(sum(contingency,1)));
    total_pairs = choose2(n);
    fp = col_pairs - tp;
    fn = row_pairs - tp;
    tn = total_pairs - tp - fp - fn;
    ri = (tp + tn) / total_pairs;
end
