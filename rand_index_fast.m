function ri = rand_index_fast(y_true, y_pred)
    y_true = y_true(:);
    y_pred = y_pred(:);
    if numel(y_true) ~= numel(y_pred)
        error('两个标签向量长度必须相同。');
    end

    [~, ~, a] = unique(y_true, 'stable');
    [~, ~, b] = unique(y_pred, 'stable');
    C = accumarray([a, b], 1);

    choose2 = @(x) x .* (x - 1) / 2;
    tp = sum(choose2(C(:)));
    row_pairs = sum(choose2(sum(C, 2)));
    col_pairs = sum(choose2(sum(C, 1)));
    total_pairs = choose2(numel(y_true));
    tn = total_pairs - row_pairs - col_pairs + tp;

    if total_pairs == 0
        ri = 1;
    else
        ri = (tp + tn) / total_pairs;
    end
end
