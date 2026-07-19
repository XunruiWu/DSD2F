function results = run_sensitivity_experiments(data_file, out_dir, opts)  % 中文说明：定义 MATLAB 函数 “run_sensitivity_experiments”并声明输入与输出。
%RUN_SENSITIVITY_EXPERIMENTS  运行 rank r 与 k-NN 参数 k 的敏感性实验。
%
% 输出 CSV 可直接用于绘制审稿回复中的敏感性图或表。

    if nargin < 3  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        opts = struct();  % 中文说明：计算并更新保存算法或实验参数的结构体。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    if ~isfield(opts, 'runs'), opts.runs = 10; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~isfield(opts, 'k_values'), opts.k_values = [5, 7, 10, 15]; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~isfield(opts, 'rank_factors'), opts.rank_factors = [0.5, 0.75, 1, 1.25, 1.5]; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~isfield(opts, 'solver'), opts.solver = struct(); end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。

    S = load(data_file);  % 中文说明：从 MAT 文件读取数据并赋给 S。
    y = local_get_labels(S);  % 中文说明：计算并更新真实标签向量。
    c = numel(unique(y));  % 中文说明：计算并更新真实类别数。
    rank_values = unique(max(1, round(c * opts.rank_factors)));  % 中文说明：计算并更新需要测试的 rank 值。

    rows = {};  % 中文说明：计算并更新暂存敏感性结果的单元数组。
    row_count = 0;  % 中文说明：计算并更新已写入结果的行数。

    % ---------- rank 敏感性：固定 k=7 ----------
    for r = rank_values  % 中文说明：开始按给定索引范围执行循环。
        sub_opts.runs = opts.runs;  % 中文说明：计算并更新当前敏感性子实验的参数。
        sub_opts.knn_k = 7;  % 中文说明：计算并更新当前敏感性子实验的参数。
        sub_opts.rank = r;  % 中文说明：计算并更新当前敏感性子实验的参数。
        sub_opts.solver = opts.solver;  % 中文说明：计算并更新当前敏感性子实验的参数。
        sub_opts.rebuild_affinity = ~isfield(S, 'A');  % 中文说明：计算并更新当前敏感性子实验的参数。
        tmp_dir = fullfile(out_dir, sprintf('rank_%d', r));  % 中文说明：计算并更新当前子实验的输出目录。
        Ssum = run_dataset_experiment(data_file, tmp_dir, sub_opts);  % 中文说明：计算并更新当前子实验返回的汇总表。

        row_count = row_count + 1;  % 中文说明：计算并更新已写入结果的行数。
        rows(row_count, :) = {'rank', r, Ssum.RI_mean, Ssum.RI_std, ...  % 中文说明：把当前敏感性实验的统计结果追加到结果单元数组。
            Ssum.NMI_mean, Ssum.NMI_std, Ssum.CPU_mean, ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
            Ssum.UnitU, Ssum.UnitV, Ssum.FallbackU, Ssum.FallbackV}; %#ok<AGROW>
    end  % 中文说明：结束当前条件、循环或函数代码块。

    % ---------- k 敏感性：固定 r=类别数 ----------
    for k = opts.k_values  % 中文说明：开始按给定索引范围执行循环。
        sub_opts.runs = opts.runs;  % 中文说明：计算并更新当前敏感性子实验的参数。
        sub_opts.knn_k = k;  % 中文说明：计算并更新当前敏感性子实验的参数。
        sub_opts.rank = c;  % 中文说明：计算并更新当前敏感性子实验的参数。
        sub_opts.solver = opts.solver;  % 中文说明：计算并更新当前敏感性子实验的参数。
        sub_opts.rebuild_affinity = true;  % 中文说明：计算并更新当前敏感性子实验的参数。
        tmp_dir = fullfile(out_dir, sprintf('knn_%d', k));  % 中文说明：计算并更新当前子实验的输出目录。
        Ssum = run_dataset_experiment(data_file, tmp_dir, sub_opts);  % 中文说明：计算并更新当前子实验返回的汇总表。

        row_count = row_count + 1;  % 中文说明：计算并更新已写入结果的行数。
        rows(row_count, :) = {'knn', k, Ssum.RI_mean, Ssum.RI_std, ...  % 中文说明：把当前敏感性实验的统计结果追加到结果单元数组。
            Ssum.NMI_mean, Ssum.NMI_std, Ssum.CPU_mean, ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
            Ssum.UnitU, Ssum.UnitV, Ssum.FallbackU, Ssum.FallbackV}; %#ok<AGROW>
    end  % 中文说明：结束当前条件、循环或函数代码块。

    results = cell2table(rows, 'VariableNames', ...  % 中文说明：把统计量整理为结果表并赋给 results。
        {'Type','Value','RI_mean','RI_std','NMI_mean','NMI_std','CPU_mean', ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
         'UnitU','UnitV','FallbackU','FallbackV'});  % 中文说明：继续补充上一行表达式或函数调用的参数。
    writetable(results, fullfile(out_dir, 'sensitivity_summary.csv'));  % 中文说明：把结果表写入 CSV 文件。
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
        error('未找到标签变量。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    y = y(:);  % 中文说明：计算并更新真实标签向量。
end  % 中文说明：结束当前条件、循环或函数代码块。
