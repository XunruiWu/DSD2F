function ri = rand_index_fast(y_true, y_pred)  % 中文说明：定义 MATLAB 函数 “rand_index_fast”并声明输入与输出。
%RAND_INDEX_FAST  用列联表计算 Rand Index，避免显式枚举所有样本对。

    y_true = y_true(:);  % 中文说明：计算右侧表达式并把结果赋给 y_true。
    y_pred = y_pred(:);  % 中文说明：计算右侧表达式并把结果赋给 y_pred。
    if numel(y_true) ~= numel(y_pred)  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        error('两个标签向量长度必须相同。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    [~, ~, a] = unique(y_true, 'stable');  % 中文说明：把任意标签重新编码为连续正整数。
    [~, ~, b] = unique(y_pred, 'stable');  % 中文说明：把任意标签重新编码为连续正整数。
    C = accumarray([a, b], 1);  % 中文说明：计算并更新真实标签与预测标签的列联表。

    choose2 = @(x) x .* (x - 1) / 2;  % 中文说明：计算右侧表达式并把结果赋给 choose2。
    tp = sum(choose2(C(:)));  % 中文说明：计算右侧表达式并把结果赋给 tp。
    row_pairs = sum(choose2(sum(C, 2)));  % 中文说明：计算右侧表达式并把结果赋给 row_pairs。
    col_pairs = sum(choose2(sum(C, 1)));  % 中文说明：计算右侧表达式并把结果赋给 col_pairs。
    total_pairs = choose2(numel(y_true));  % 中文说明：计算右侧表达式并把结果赋给 total_pairs。
    tn = total_pairs - row_pairs - col_pairs + tp;  % 中文说明：计算右侧表达式并把结果赋给 tn。

    if total_pairs == 0  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        ri = 1;  % 中文说明：计算并更新Rand 指数。
    else  % 中文说明：进入前述条件不成立时的分支。
        ri = (tp + tn) / total_pairs;  % 中文说明：计算并更新Rand 指数。
    end  % 中文说明：结束当前条件、循环或函数代码块。
end  % 中文说明：结束当前条件、循环或函数代码块。
