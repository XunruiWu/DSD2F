function plot_dsd2f_history(history, out_dir, prefix)  % 中文说明：定义 MATLAB 函数并声明输入与输出。
%PLOT_DSD2F_HISTORY  输出论文重跑时常用的收敛曲线。

    if nargin < 2 || isempty(out_dir), out_dir = pwd; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if nargin < 3 || isempty(prefix), prefix = 'dsd2f'; end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。

    it = (1:history.iterations)';  % 中文说明：计算并更新绘图使用的迭代序号。

    figure;  % 中文说明：新建一个绘图窗口。
    semilogy(it, max(history.model_obj, eps), 'LineWidth', 1.5);  % 中文说明：使用纵轴对数坐标绘制收敛曲线。
    xlabel('Iterations'); ylabel('Model objective'); grid on;  % 中文说明：设置当前图形的横轴标签。
    saveas(gcf, fullfile(out_dir, [prefix '_objective.png']));  % 中文说明：把当前图形保存到指定文件。

    figure;  % 中文说明：新建一个绘图窗口。
    semilogy(it, max(history.row_residual, eps), 'LineWidth', 1.5);  % 中文说明：使用纵轴对数坐标绘制收敛曲线。
    xlabel('Iterations'); ylabel('Row-sum residual'); grid on;  % 中文说明：设置当前图形的横轴标签。
    saveas(gcf, fullfile(out_dir, [prefix '_rowsum.png']));  % 中文说明：把当前图形保存到指定文件。

    figure;  % 中文说明：新建一个绘图窗口。
    semilogy(it, max(history.factor_residual, eps), 'LineWidth', 1.5);  % 中文说明：使用纵轴对数坐标绘制收敛曲线。
    xlabel('Iterations'); ylabel('Factor residual'); grid on;  % 中文说明：设置当前图形的横轴标签。
    saveas(gcf, fullfile(out_dir, [prefix '_factor.png']));  % 中文说明：把当前图形保存到指定文件。
end  % 中文说明：结束当前条件、循环或函数代码块。
