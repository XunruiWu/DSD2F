function value = dsd2f_augmented_value(A, A_fro2, U, V, Lambda, gamma, eta)
    n = size(U, 1);
    one = ones(n, 1);

    UtU = U' * U;
    VtV = V' * V;
    AV = A * V;

    fit2 = A_fro2 + trace(UtU * VtV) - 2 * sum(sum(U .* AV));
    fit2 = max(real(fit2), 0);

    residual = U * (V' * one) - one;
    consistency = norm(U - V, 'fro')^2;

    value = 0.5 * fit2 ...
          + Lambda' * residual ...
          + 0.5 * eta * (residual' * residual) ...
          + 0.5 * gamma * consistency;
    value = real(value);
end

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

function ri = rand_index_fast(y_true, y_pred)
    y_true = y_true(:);
    y_pred = y_pred(:);
    if numel(y_true) ~= numel(y_pred)
        error('Two label vectors must have same length.');
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

function summary = run_dataset_experiment(data_file, out_dir, opts)
    if nargin < 3
        opts = struct();
    end
    if ~isfield(opts, 'runs'), opts.runs = 30; end
    if ~isfield(opts, 'knn_k'), opts.knn_k = 7; end
    if ~isfield(opts, 'rank'), opts.rank = []; end
    if ~isfield(opts, 'solver'), opts.solver = struct(); end
    if ~isfield(opts, 'rebuild_affinity'), opts.rebuild_affinity = false; end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    S = load(data_file);
    y = local_get_labels(S);

    if opts.rebuild_affinity
        if ~isfield(S, 'X')
            error('Sensitivity experiment requires raw sample matrix X in MAT file.');
        end
        A = make_knn_affinity(S.X, opts.knn_k);
    elseif isfield(S, 'A')
        A = sparse(S.A);
    elseif isfield(S, 'X')
        A = make_knn_affinity(S.X, opts.knn_k);
    else
        error('MAT file must contain either A or X.');
    end

    if isempty(opts.rank)
        r = numel(unique(y));
    else
        r = opts.rank;
    end

    RI = nan(opts.runs, 1);
    NMI = nan(opts.runs, 1);
    CPU = nan(opts.runs, 1);
    UNIT_U = nan(opts.runs, 1);
    UNIT_V = nan(opts.runs, 1);
    FB_U = nan(opts.runs, 1);
    FB_V = nan(opts.runs, 1);

    histories = cell(opts.runs, 1);
    for run = 1:opts.runs
        solver_opts = opts.solver;
        solver_opts.seed = run;

        t0 = tic;
        [pred, ~, ~, ~, hist] = dsd2f_psg_fallback(A, r, solver_opts);
        CPU(run) = toc(t0);

        RI(run) = rand_index_fast(y, pred);
        NMI(run) = nmi_score(y, pred);
        UNIT_U(run) = hist.unit_accept_rate_U;
        UNIT_V(run) = hist.unit_accept_rate_V;
        FB_U(run) = hist.fallback_rate_U;
        FB_V(run) = hist.fallback_rate_V;
        histories{run} = hist;

        fprintf('run %d/%d: RI=%.4f, NMI=%.4f, time=%.3fs\n', ...
            run, opts.runs, RI(run), NMI(run), CPU(run));
    end

    summary = table( ...
        mean(RI), std(RI), mean(NMI), std(NMI), mean(CPU), std(CPU), ...
        mean(UNIT_U), mean(UNIT_V), mean(FB_U), mean(FB_V), ...
        'VariableNames', {'RI_mean','RI_std','NMI_mean','NMI_std', ...
        'CPU_mean','CPU_std','UnitU','UnitV','FallbackU','FallbackV'});

    [~, name] = fileparts(data_file);
    writetable(summary, fullfile(out_dir, [name '_summary.csv']));
    save(fullfile(out_dir, [name '_all_runs.mat']), ...
        'RI', 'NMI', 'CPU', 'UNIT_U', 'UNIT_V', 'FB_U', 'FB_V', 'histories', 'summary');
end

function y = local_get_labels(S)
    candidates = {'gnd', 'labels', 'y', 'truth'};
    y = [];
    for i = 1:numel(candidates)
        if isfield(S, candidates{i})
            y = S.(candidates{i});
            break;
        end
    end
    if isempty(y)
        error('Label variable not found; use gnd, labels, y, or truth.');
    end
    y = y(:);
end
