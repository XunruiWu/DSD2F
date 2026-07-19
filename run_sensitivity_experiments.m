function results = run_sensitivity_experiments(data_file, out_dir, opts)
    if nargin < 3
        opts = struct();
    end
    if ~isfield(opts, 'runs'), opts.runs = 10; end
    if ~isfield(opts, 'k_values'), opts.k_values = [5, 7, 10, 15]; end
    if ~isfield(opts, 'rank_factors'), opts.rank_factors = [0.5, 0.75, 1, 1.25, 1.5]; end
    if ~isfield(opts, 'solver'), opts.solver = struct(); end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    S = load(data_file);
    y = local_get_labels(S);
    c = numel(unique(y));
    rank_values = unique(max(1, round(c * opts.rank_factors)));

    rows = {};
    row_count = 0;

    for r = rank_values
        sub_opts.runs = opts.runs;
        sub_opts.knn_k = 7;
        sub_opts.rank = r;
        sub_opts.solver = opts.solver;
        sub_opts.rebuild_affinity = ~isfield(S, 'A');
        tmp_dir = fullfile(out_dir, sprintf('rank_%d', r));
        Ssum = run_dataset_experiment(data_file, tmp_dir, sub_opts);

        row_count = row_count + 1;
        rows(row_count, :) = {'rank', r, Ssum.RI_mean, Ssum.RI_std, ...
            Ssum.NMI_mean, Ssum.NMI_std, Ssum.CPU_mean, ...
            Ssum.UnitU, Ssum.UnitV, Ssum.FallbackU, Ssum.FallbackV};
    end

    for k = opts.k_values
        sub_opts.runs = opts.runs;
        sub_opts.knn_k = k;
        sub_opts.rank = c;
        sub_opts.solver = opts.solver;
        sub_opts.rebuild_affinity = true;
        tmp_dir = fullfile(out_dir, sprintf('knn_%d', k));
        Ssum = run_dataset_experiment(data_file, tmp_dir, sub_opts);

        row_count = row_count + 1;
        rows(row_count, :) = {'knn', k, Ssum.RI_mean, Ssum.RI_std, ...
            Ssum.NMI_mean, Ssum.NMI_std, Ssum.CPU_mean, ...
            Ssum.UnitU, Ssum.UnitV, Ssum.FallbackU, Ssum.FallbackV};
    end

    results = cell2table(rows, 'VariableNames', ...
        {'Type','Value','RI_mean','RI_std','NMI_mean','NMI_std','CPU_mean', ...
         'UnitU','UnitV','FallbackU','FallbackV'});
    writetable(results, fullfile(out_dir, 'sensitivity_summary.csv'));
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
        error('Label variable not found.');
    end
    y = y(:);
end
