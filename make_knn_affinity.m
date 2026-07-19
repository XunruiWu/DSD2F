function A = make_knn_affinity(X, k, opts)  % 中文说明：定义 MATLAB 函数 “make_knn_affinity”并声明输入与输出。
%MAKE_KNN_AFFINITY  构造对称二值 k-NN affinity matrix。
%
% 输入 X 的每一行是一条样本。程序先进行 l2 归一化，再寻找近邻。
% 若安装 Statistics and Machine Learning Toolbox，则优先调用 knnsearch；
% 否则采用分块距离计算，避免一次生成完整 n×n 距离矩阵。

    if nargin < 3  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        opts = struct();  % 中文说明：计算并更新保存算法或实验参数的结构体。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    if ~isfield(opts, 'block_size'), opts.block_size = 1000; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。

    [n, ~] = size(X);  % 中文说明：读取输入矩阵的行数和列数。
    if k < 1 || k >= n  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        error('k 必须满足 1 <= k < 样本数。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    % 每行归一化；零向量保持为零。
    row_norm = sqrt(sum(X.^2, 2));  % 中文说明：计算并更新每个样本的二范数。
    row_norm(row_norm == 0) = 1;  % 中文说明：把零范数替换为 1，避免归一化除零。
    X = X ./ row_norm;  % 中文说明：计算并更新样本矩阵。

    if exist('knnsearch', 'file') == 2  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        % 多取一个邻居用于排除样本自身。
        idx = knnsearch(X, X, 'K', k + 1);  % 中文说明：计算并更新每个样本的近邻下标。
        idx = idx(:, 2:end);  % 中文说明：计算并更新每个样本的近邻下标。
    else  % 中文说明：进入前述条件不成立时的分支。
        idx = zeros(n, k);  % 中文说明：创建全零数组并赋给 idx。
        xnorm2 = sum(X.^2, 2)';  % 中文说明：计算右侧表达式并把结果赋给 xnorm2。
        for first = 1:opts.block_size:n  % 中文说明：开始按给定索引范围执行循环。
            last = min(n, first + opts.block_size - 1);  % 中文说明：计算右侧表达式并把结果赋给 last。
            Xb = X(first:last, :);  % 中文说明：计算右侧表达式并把结果赋给 Xb。
            bnorm2 = sum(Xb.^2, 2);  % 中文说明：计算右侧表达式并把结果赋给 bnorm2。
            D = bnorm2 + xnorm2 - 2 * (Xb * X');  % 中文说明：计算并更新当前数据块到所有样本的距离平方矩阵。
            D = max(D, 0);  % 中文说明：计算并更新当前数据块到所有样本的距离平方矩阵。

            % 排除自身。
            rows = (first:last)';  % 中文说明：计算并更新暂存敏感性结果的单元数组。
            local_rows = (1:numel(rows))';  % 中文说明：计算右侧表达式并把结果赋给 local_rows。
            D(sub2ind(size(D), local_rows, rows)) = inf;  % 中文说明：修改距离矩阵中的指定元素。

            [~, block_idx] = mink(D, k, 2);  % 中文说明：在每一行中找出距离最小的 k 个近邻。
            idx(first:last, :) = block_idx;  % 中文说明：把当前数据块的近邻下标写入总索引矩阵。
        end  % 中文说明：结束当前条件、循环或函数代码块。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    row_id = repelem((1:n)', k, 1);  % 中文说明：计算并更新稀疏矩阵中边的行下标。
    col_id = reshape(idx', [], 1);  % 中文说明：计算并更新稀疏矩阵中边的列下标。
    A0 = sparse(row_id, col_id, 1, n, n);  % 中文说明：构造或转换为稀疏矩阵并赋给 A0。

    % 与论文一致：A = max(A0, A0^T)，并清除对角线。
    A = spones(max(A0, A0'));  % 中文说明：计算并更新亲和矩阵。
    A = A - spdiags(diag(A), 0, n, n);  % 中文说明：计算并更新亲和矩阵。
end  % 中文说明：结束当前条件、循环或函数代码块。
