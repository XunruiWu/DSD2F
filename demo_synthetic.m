%DEMO_SYNTHETIC  小规模合成数据演示。
% 先运行该脚本检查代码路径和基本数值行为，再运行正式数据集。

rng(1);  % 中文说明：设置随机数种子以保证实验可复现。
n_per_cluster = 40;  % 中文说明：计算并更新每个合成簇的样本数。
c = 3;  % 中文说明：计算并更新真实类别数。
d = 5;  % 中文说明：计算并更新特征维数。
X = [];  % 中文说明：计算并更新样本矩阵。
y = [];  % 中文说明：计算并更新真实标签向量。
for j = 1:c  % 中文说明：开始按给定索引范围执行循环。
    center = zeros(1, d);  % 中文说明：创建全零数组并赋给 center。
    center(j) = 4;  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
    X = [X; randn(n_per_cluster, d) + center]; %#ok<AGROW>
    y = [y; j * ones(n_per_cluster, 1)]; %#ok<AGROW>
end  % 中文说明：结束当前条件、循环或函数代码块。

A = make_knn_affinity(X, 7);  % 中文说明：根据样本构造 kNN 亲和矩阵并赋给 A。
opts.verbose = true;  % 中文说明：计算并更新保存算法或实验参数的结构体。
opts.max_iter = 500;  % 中文说明：计算并更新保存算法或实验参数的结构体。
opts.seed = 1;  % 中文说明：计算并更新保存算法或实验参数的结构体。
opts.eta = 1;  % 中文说明：计算并更新保存算法或实验参数的结构体。

[pred, U, V, Lambda, history] = dsd2f_psg_fallback(A, c, opts); %#ok<ASGLU>
fprintf('RI  = %.4f\n', rand_index_fast(y, pred));  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
fprintf('NMI = %.4f\n', nmi_score(y, pred));  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
fprintf('U unit-step acceptance = %.3f\n', history.unit_accept_rate_U);  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
fprintf('V unit-step acceptance = %.3f\n', history.unit_accept_rate_V);  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
fprintf('U fallback rate = %.3f\n', history.fallback_rate_U);  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
fprintf('V fallback rate = %.3f\n', history.fallback_rate_V);  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。

plot_dsd2f_history(history, pwd, 'synthetic');  % 中文说明：调用绘图函数输出本次运行的收敛曲线。
