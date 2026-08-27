function summary = run_dsd2f_table910()
%RUN_DSD2F_TABLE910 Reproduce the DSD2F columns of Tables 9 and 10.
%
% Each data/<DATASET>.mat file must contain:
%   A   : prepared symmetric affinity matrix
%   gnd : ground-truth labels (used only after optimization for RI)
%
% Solver time excludes data loading, initialization generation and k-means.

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root,'src'));
    addpath(fullfile(root,'utils'));

    cfg = table910_config();
    opts = dsd2f_options();
    raw_dir = fullfile(root,'results','raw');
    summary_dir = fullfile(root,'results','summary');
    if ~exist(raw_dir,'dir'), mkdir(raw_dir); end
    if ~exist(summary_dir,'dir'), mkdir(summary_dir); end

    rows = cell(numel(cfg.datasets)*numel(cfg.seeds), 9);
    row_id = 0;

    for d = 1:numel(cfg.datasets)
        dataset = cfg.datasets{d};
        file_path = fullfile(root,'data',[dataset '.mat']);
        if ~exist(file_path,'file')
            error('Missing data file: %s. See data/README.md.', file_path);
        end
        loaded = load(file_path);
        if ~isfield(loaded,'A') || ~isfield(loaded,'gnd')
            error('%s must contain variables A and gnd.', file_path);
        end

        A = sparse(double(loaded.A));
        gnd = loaded.gnd(:);
        n = size(A,1);
        r = cfg.classes(d);
        if n ~= cfg.samples(d) || size(A,2) ~= n
            error('%s has size %d; expected %d.', dataset, n, cfg.samples(d));
        end
        if numel(gnd) ~= n
            error('%s: gnd length does not match A.', dataset);
        end
        asym = norm(A-A','fro') / max(1,norm(A,'fro'));
        if asym > 1e-12
            error('%s: A must be symmetric for the frozen V4 run.', dataset);
        end

        fprintf('\n[%s] n=%d r=%d\n', dataset, n, r);
        for seed = cfg.seeds
            rng(seed,'twister');
            Z0 = max(randn(n,r),0) + 1e-12;

            timer = tic;
            [F, info] = dsd2f(A, r, Z0, opts);
            solver_time = toc(timer);

            % Common decoder is intentionally outside solver timing.
            rng(seed,'twister');
            embedding = normalize_rows(F);
            labels = kmeans(embedding, r, 'Start','plus', ...
                'Replicates',cfg.kmeans_replicates, ...
                'MaxIter',cfg.kmeans_max_iter, 'Display','off');
            RI = rand_index(gnd, labels);

            row_id = row_id + 1;
            rows(row_id,:) = {dataset,seed,n,r,RI,solver_time, ...
                info.iterations,info.final_gamma,info.final_balanced_row_residual};

            fprintf('  seed %2d: RI=%7.4f  time=%8.4fs  iter=%4d\n', ...
                seed,100*RI,solver_time,info.iterations);
        end
    end

    raw = cell2table(rows, 'VariableNames', ...
        {'Dataset','Seed','Samples','Rank','RI','SolverTime', ...
         'Iterations','FinalGamma','FinalBalancedRowResidual'});
    writetable(raw, fullfile(raw_dir,'dsd2f_table9_10_all_runs.csv'));

    summary = aggregate_dsd2f_table910(raw, cfg);
    writetable(summary, fullfile(summary_dir,'dsd2f_table9_10_summary.csv'));

    table9 = summary(:,{'Dataset','RI_mean_percent','RI_std_percent'});
    table10 = summary(:,{'Dataset','SolverTime_mean_s','SolverTime_std_s'});
    writetable(table9, fullfile(summary_dir,'table9_dsd2f.csv'));
    writetable(table10, fullfile(summary_dir,'table10_dsd2f.csv'));

    fprintf('\nSummary written to results/summary/.\n');
end
