function opts = dsd2f_options()
%DSD2F_OPTIONS Frozen settings for the V4 Table 9/10 DSD2F rerun.

    opts = struct();
    opts.eta = 50;
    opts.gamma0 = 1e-5;
    opts.gamma_min = 1e-4;
    opts.gamma_max = 1e3;
    opts.eps_tau = 1e-12;
    opts.max_iter = 3000;
    opts.tol = 1e-6;

    % Low-rank balancing settings.
    opts.balance_eps = 1e-12;
    opts.balance_tol = 1e-10;
    opts.balance_max_iter = 500;
end
