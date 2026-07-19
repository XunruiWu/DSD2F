function plot_dsd2f_history(history, out_dir, prefix)
    if nargin < 2 || isempty(out_dir), out_dir = pwd; end
    if nargin < 3 || isempty(prefix), prefix = 'dsd2f'; end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    it = (1:history.iterations)';

    figure;
    semilogy(it, max(history.model_obj, eps), 'LineWidth', 1.5);
    xlabel('Iterations'); ylabel('Model objective'); grid on;
    saveas(gcf, fullfile(out_dir, [prefix '_objective.png']));

    figure;
    semilogy(it, max(history.row_residual, eps), 'LineWidth', 1.5);
    xlabel('Iterations'); ylabel('Row-sum residual'); grid on;
    saveas(gcf, fullfile(out_dir, [prefix '_rowsum.png']));

    figure;
    semilogy(it, max(history.factor_residual, eps), 'LineWidth', 1.5);
    xlabel('Iterations'); ylabel('Factor residual'); grid on;
    saveas(gcf, fullfile(out_dir, [prefix '_factor.png']));
end
