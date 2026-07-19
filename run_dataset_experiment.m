function summary = run_dataset_experiment(data_file, out_dir, opts)  % 中文说明：定义 MATLAB 函数 “run_dataset_experiment”并声明输入与输出。
%RUN_DATASET_EXPERIMENT  对单个数据集重复运行 DSD2F 并保存统计结果。
%
% 数据文件支持两种形式：
%   1) 直接包含 affinity matrix A；
%   2) 包含样本矩阵 X，程序按 opts.knn_k 构造 k-NN 图。
% 标签变量可命名为 gnd、labels、y 或 truth。

    if nargin < 3  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        opts = struct();  % 中文说明：计算并更新保存算法或实验参数的结构体。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    if ~isfield(opts, 'runs'), opts.runs = 30; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~isfield(opts, 'knn_k'), opts.knn_k = 7; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~isfield(opts, 'rank'), opts.rank = []; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~isfield(opts, 'solver'), opts.solver = struct(); end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~isfield(opts, 'rebuild_affinity'), opts.rebuild_affinity = false; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。

    S = load(data_file);  % 中文说明：从 MAT 文件读取数据并赋给 S。
    y = local_get_labels(S);  % 中文说明：计算并更新真实标签向量。

    if opts.rebuild_affinity  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        if ~isfield(S, 'X')  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
            error('k 敏感性实验需要 MAT 文件包含原始样本矩阵 X。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
        end  % 中文说明：结束当前条件、循环或函数代码块。
        A = make_knn_affinity(S.X, opts.knn_k);  % 中文说明：根据样本构造 kNN 亲和矩阵并赋给 A。
    elseif isfield(S, 'A')  % 中文说明：在前一条件不成立时继续判断这一条件。
        A = sparse(S.A);  % 中文说明：构造或转换为稀疏矩阵并赋给 A。
    elseif isfield(S, 'X')  % 中文说明：在前一条件不成立时继续判断这一条件。
        A = make_knn_affinity(S.X, opts.knn_k);  % 中文说明：根据样本构造 kNN 亲和矩阵并赋给 A。
    else  % 中文说明：进入前述条件不成立时的分支。
        error('MAT 文件中必须包含 A 或 X。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    if isempty(opts.rank)  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        r = numel(unique(y));  % 中文说明：计算并更新低秩因子列数。
    else  % 中文说明：进入前述条件不成立时的分支。
        r = opts.rank;  % 中文说明：计算并更新低秩因子列数。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    RI = nan(opts.runs, 1);  % 中文说明：预分配 NaN 数组以保存 RI。
    NMI = nan(opts.runs, 1);  % 中文说明：预分配 NaN 数组以保存 NMI。
    CPU = nan(opts.runs, 1);  % 中文说明：预分配 NaN 数组以保存 CPU。
    UNIT_U = nan(opts.runs, 1);  % 中文说明：预分配 NaN 数组以保存 UNIT_U。
    UNIT_V = nan(opts.runs, 1);  % 中文说明：预分配 NaN 数组以保存 UNIT_V。
    FB_U = nan(opts.runs, 1);  % 中文说明：预分配 NaN 数组以保存 FB_U。
    FB_V = nan(opts.runs, 1);  % 中文说明：预分配 NaN 数组以保存 FB_V。

    histories = cell(opts.runs, 1);  % 中文说明：计算并更新保存每次运行历史的单元数组。
    for run = 1:opts.runs  % 中文说明：开始按给定索引范围执行循环。
        solver_opts = opts.solver;  % 中文说明：计算并更新传给主求解器的参数。
        solver_opts.seed = run;  % 中文说明：计算并更新传给主求解器的参数。

        t0 = tic;  % 中文说明：启动计时器以统计运行时间。
        [pred, ~, ~, ~, hist] = dsd2f_psg_fallback(A, r, solver_opts);  % 中文说明：调用 DSD2F 主函数得到标签、因子、乘子和历史记录。
        CPU(run) = toc(t0);  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。

        RI(run) = rand_index_fast(y, pred);  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
        NMI(run) = nmi_score(y, pred);  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
        UNIT_U(run) = hist.unit_accept_rate_U;  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
        UNIT_V(run) = hist.unit_accept_rate_V;  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
        FB_U(run) = hist.fallback_rate_U;  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
        FB_V(run) = hist.fallback_rate_V;  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
        histories{run} = hist;  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。

        fprintf('run %d/%d: RI=%.4f, NMI=%.4f, time=%.3fs\n', ...  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
            run, opts.runs, RI(run), NMI(run), CPU(run));  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    summary = table( ...  % 中文说明：把统计量整理为结果表并赋给 summary。
        mean(RI), std(RI), mean(NMI), std(NMI), mean(CPU), std(CPU), ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
        mean(UNIT_U), mean(UNIT_V), mean(FB_U), mean(FB_V), ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
        'VariableNames', {'RI_mean','RI_std','NMI_mean','NMI_std', ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
        'CPU_mean','CPU_std','UnitU','UnitV','FallbackU','FallbackV'});  % 中文说明：继续补充上一行表达式或函数调用的参数。

    [~, name] = fileparts(data_file);  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
    writetable(summary, fullfile(out_dir, [name '_summary.csv']));  % 中文说明：把结果表写入 CSV 文件。
    save(fullfile(out_dir, [name '_all_runs.mat']), ...  % 中文说明：把当前实验变量保存到 MAT 文件。
        'RI', 'NMI', 'CPU', 'UNIT_U', 'UNIT_V', 'FB_U', 'FB_V', 'histories', 'summary');  % 中文说明：继续补充上一行表达式或函数调用的参数。
end  % 中文说明：结束当前条件、循环或函数代码块。

function y = local_get_labels(S)  % 中文说明：定义 MATLAB 函数 “local_get_labels”并声明输入与输出。
    candidates = {'gnd', 'labels', 'y', 'truth'};  % 中文说明：计算右侧表达式并把结果赋给 candidates。
    y = [];  % 中文说明：计算并更新真实标签向量。
    for i = 1:numel(candidates)  % 中文说明：开始按给定索引范围执行循环。
        if isfield(S, candidates{i})  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
            y = S.(candidates{i});  % 中文说明：计算并更新真实标签向量。
            break;  % 中文说明：满足条件后提前退出当前循环。
        end  % 中文说明：结束当前条件、循环或函数代码块。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    if isempty(y)  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        error('未找到标签变量；请使用 gnd、labels、y 或 truth。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    y = y(:);  % 中文说明：计算并更新真实标签向量。
end  % 中文说明：结束当前条件、循环或函数代码块。
