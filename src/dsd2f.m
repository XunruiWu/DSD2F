function [F, info] = dsd2f(A, r, Z0, opts)
%DSD2F Low-rank two-factor doubly stochastic clustering solver.
%
%   [F,INFO] = DSD2F(A,R,Z0,OPTS) runs the numerical configuration used
%   for the additional V4 experiment reported in Tables 9 and 10.
%
%   Inputs
%   ------
%   A    : n-by-n nonnegative symmetric affinity matrix (preferably sparse)
%   r    : factor rank
%   Z0   : n-by-r nonnegative initialization; U0 = V0 = Z0
%   opts : options structure returned by dsd2f_options()
%
%   Outputs
%   -------
%   F    : n-by-r final low-rank embedding
%   info : iteration count, penalty trajectory and feasibility diagnostics
%
%   The implementation never forms an n-by-n factor product. One shared
%   low-rank balancing correction is applied after each U/V sweep, and a
%   high-accuracy low-rank symmetric balancing is applied at termination.

    if nargin < 4 || isempty(opts)
        opts = dsd2f_options();
    end

    A = sparse(double(A));
    [n, m] = size(A);
    if n ~= m
        error('DSD2F:AffinityNotSquare', 'A must be square.');
    end
    if size(Z0,1) ~= n || size(Z0,2) ~= r
        error('DSD2F:BadInitialization', 'Z0 must have size n-by-r.');
    end
    if any(nonzeros(A) < 0)
        error('DSD2F:NegativeAffinity', 'A must be nonnegative.');
    end

    one = ones(n,1);
    Ir = eye(r);
    U = max(double(Z0), 0);
    V = U;
    Lambda = zeros(n,1);
    gamma = opts.gamma0;
    eta = opts.eta;
    A_fro2 = full(sum(nonzeros(A).^2));
    previous_objective = NaN;

    history.objective = nan(opts.max_iter,1);
    history.gamma = nan(opts.max_iter,1);
    history.uv_row_residual = nan(opts.max_iter,1);
    history.symmetric_row_residual = nan(opts.max_iter,1);
    history.factor_gap = nan(opts.max_iter,1);
    history.relative_objective_change = nan(opts.max_iter,1);

    for k = 1:opts.max_iter
        % U update.
        sumV = one' * V;
        numeratorU = gamma * V + A * V + (eta * one - Lambda) * sumV;
        denominatorU = gamma * Ir + V' * V + eta * (sumV' * sumV);
        U = max(numeratorU / denominatorU, 0);

        % V update. The stored V4 affinity matrices are symmetric.
        sumU = one' * U;
        AU = A * U;
        gramU = U' * U;
        numeratorV = gamma * U + AU + (eta * one - Lambda) * sumU;
        denominatorV = gamma * Ir + gramU + eta * (sumU' * sumU);
        V = max(numeratorV / denominatorV, 0);

        % One shared low-rank feasibility correction after the full sweep.
        [U, V] = shared_balance_once(U, V, opts.balance_eps);

        % KKT-motivated multiplier recovery on the corrected factors.
        gramV = V' * V;
        multiplier_core = A * V - U * gramV + gamma * (V - U);
        Lambda = sum(multiplier_core .* U, 2);

        % Low-rank reconstruction quantity used by the V4 stopping rule.
        F0 = 0.5 * (U + V);
        gramF = F0' * F0;
        objective = 0.5 * (sum(sum(gramF .* gramF)) + A_fro2) ...
                    - sum(sum(F0 .* (A * F0)));
        history.objective(k) = objective;

        uv_residual = U * (V' * one) - one;
        symmetric_rows = 0.5 * (U * (V' * one) + V * (U' * one));
        history.uv_row_residual(k) = norm(uv_residual, 2) / sqrt(n);
        history.symmetric_row_residual(k) = norm(symmetric_rows - one, 2) / sqrt(n);
        history.factor_gap(k) = norm(U - V, 'fro') / ...
            max(1, norm(U, 'fro') + norm(V, 'fro'));

        % Alignment-based adaptive penalty, clipped to the V4 interval.
        innerUV = sum(U(:) .* V(:));
        denominator = 2 * max(abs(innerUV), opts.eps_tau);
        ratio = (sum(U(:).^2) + sum(V(:).^2)) / denominator;
        gamma = min(opts.gamma_max, max(opts.gamma_min, gamma * max(1, ratio)));
        history.gamma(k) = gamma;

        if k > 1
            relative_change = abs(objective - previous_objective) / ...
                              (abs(previous_objective) + 1);
            history.relative_objective_change(k) = relative_change;
            if relative_change < opts.tol
                break;
            end
        end
        previous_objective = objective;
    end

    % Terminal high-accuracy balancing of the implicit symmetric consensus.
    [d, balance_info] = final_symmetric_balance( ...
        U, V, opts.balance_eps, opts.balance_tol, opts.balance_max_iter);
    F = 0.5 * (U + V) .* d;

    fields = fieldnames(history);
    for i = 1:numel(fields)
        history.(fields{i}) = history.(fields{i})(1:k,:);
    end

    info = struct();
    info.iterations = k;
    info.final_objective = history.objective(end);
    info.final_gamma = gamma;
    info.uv_row_residual = history.uv_row_residual(end);
    info.symmetric_row_residual_pre_final = history.symmetric_row_residual(end);
    info.final_balanced_row_residual = balance_info.residual;
    info.final_balance_iterations = balance_info.iterations;
    info.factor_gap = history.factor_gap(end);
    info.history = history;
end
