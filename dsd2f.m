function [labels, U, V, Lambda, history] = dsd2f(A, r, opts)
    if nargin < 3, opts = struct(); end
    opts = local_defaults(opts);

    [n, m] = size(A);
    if n ~= m, error('A must be square.'); end
    if r < 1 || r ~= floor(r), error('r must be a positive integer.'); end
    if any(~isfinite(nonzeros(A))), error('A contains NaN or Inf.'); end
    if any(nonzeros(A) < 0), error('A must be a nonnegative affinity matrix.'); end

    one = ones(n, 1);
    Ir = eye(r);
    rng(opts.seed, 'twister');

    if isempty(opts.U0)
        U = max(randn(n, r), 0);
    else
        U = max(opts.U0, 0);
    end
    if isempty(opts.V0)
        V = max(randn(n, r), 0);
    else
        V = max(opts.V0, 0);
    end
    U = U + opts.init_floor;
    V = V + opts.init_floor;
    Lambda = zeros(n, 1);
    gamma = opts.gamma0;

    A_fro2 = full(sum(nonzeros(A).^2));
    T = opts.max_iter;

    history.model_obj = nan(T, 1);
    history.aug_obj_before_dual = nan(T, 1);
    history.row_residual = nan(T, 1);
    history.factor_residual = nan(T, 1);
    history.rel_U = nan(T, 1);
    history.rel_V = nan(T, 1);
    history.gamma = nan(T, 1);
    history.alpha_U = nan(T, 1);
    history.alpha_V = nan(T, 1);
    history.unit_accept_U = false(T, 1);
    history.unit_accept_V = false(T, 1);
    history.fallback_U = false(T, 1);
    history.fallback_V = false(T, 1);
    history.time = nan(T, 1);

    tic_total = tic;

    for k = 1:T
        U_old = U;
        V_old = V;

        L_start = dsd2f_augmented_value(A, A_fro2, U, V, Lambda, gamma, opts.eta);

        s = V' * one;
        MU = V' * V + opts.eta * (s * s') + gamma * Ir;
        GU = A * V + opts.eta * (one * s') - Lambda * s' + gamma * V;
        gradU = U * MU - GU;

        LU = max(eig((MU + MU') / 2));
        LU = max(real(LU), opts.min_curvature);

        alphaU = 1;
        U_candidate = max(U - alphaU * (gradU / MU), 0);
        dU2 = norm(U_candidate - U, 'fro')^2;
        L_candidate = dsd2f_augmented_value(A, A_fro2, U_candidate, V, Lambda, gamma, opts.eta);

        if L_candidate <= L_start - opts.armijo_c * dU2 + opts.numeric_tol
            U = U_candidate;
            acceptedU = true;
            history.alpha_U(k) = alphaU;
        else
            tU = opts.theta / LU;
            U = max(U_old - tU * gradU, 0);
            acceptedU = false;
            history.alpha_U(k) = tU;
            history.fallback_U(k) = true;
        end
        history.unit_accept_U(k) = acceptedU;

        R = U' * U;
        MV = (1 + opts.eta * n) * R + gamma * Ir;

        VR_minus_U = V * R - U;
        gradV = V * R - A' * U ...
              + one * (Lambda' * U) ...
              + opts.eta * one * sum(VR_minus_U, 1) ...
              + gamma * (V - U);

        LV = (1 + opts.eta * n) * max(eig((R + R') / 2)) + gamma;
        LV = max(real(LV), opts.min_curvature);

        L_after_U = dsd2f_augmented_value(A, A_fro2, U, V, Lambda, gamma, opts.eta);
        alphaV = 1;
        V_candidate = max(V - alphaV * (gradV / MV), 0);
        dV2 = norm(V_candidate - V, 'fro')^2;
        L_candidate = dsd2f_augmented_value(A, A_fro2, U, V_candidate, Lambda, gamma, opts.eta);

        if L_candidate <= L_after_U - opts.armijo_c * dV2 + opts.numeric_tol
            V = V_candidate;
            acceptedV = true;
            history.alpha_V(k) = alphaV;
        else
            tV = opts.theta / LV;
            V = max(V_old - tV * gradV, 0);
            acceptedV = false;
            history.alpha_V(k) = tV;
            history.fallback_V(k) = true;
        end
        history.unit_accept_V(k) = acceptedV;

        AV = A * V;
        VtV = V' * V;
        multiplier_core = AV - U * VtV + gamma * (V - U);
        Lambda_new = sum(multiplier_core .* U, 2);

        innerUV = sum(U(:) .* V(:));
        ratio = (norm(U, 'fro')^2 + norm(V, 'fro')^2) / ...
                (2 * max(abs(innerUV), opts.eps_tau));
        tau = max(1, ratio);
        gamma_new = min(opts.gamma_max, gamma * tau);

        relU = norm(U - U_old, 'fro') / max(1, norm(U_old, 'fro'));
        relV = norm(V - V_old, 'fro') / max(1, norm(V_old, 'fro'));
        row_res = U * (V' * one) - one;
        feas = norm(row_res, 2) / sqrt(n);
        factor_res = norm(U - V, 'fro') / max(1, norm(U, 'fro') + norm(V, 'fro'));

        history.model_obj(k) = dsd2f_model_value(A, A_fro2, U, V, gamma);
        history.aug_obj_before_dual(k) = dsd2f_augmented_value(A, A_fro2, U, V, Lambda, gamma, opts.eta);
        history.row_residual(k) = feas;
        history.factor_residual(k) = factor_res;
        history.rel_U(k) = relU;
        history.rel_V(k) = relV;
        history.gamma(k) = gamma_new;
        history.time(k) = toc(tic_total);

        Lambda = Lambda_new;
        gamma = gamma_new;

        if opts.verbose && (k == 1 || mod(k, opts.print_every) == 0)
            fprintf(['iter=%4d  model=%.6e  feas=%.3e  relU=%.3e  relV=%.3e  ' ...
                     'gamma=%.3e  fbU=%d  fbV=%d\n'], ...
                k, history.model_obj(k), feas, relU, relV, gamma, ...
                history.fallback_U(k), history.fallback_V(k));
        end

        if max(relU, relV) <= opts.tol_step && feas <= opts.tol_feas
            break;
        end
    end

    fields = fieldnames(history);
    for i = 1:numel(fields)
        value = history.(fields{i});
        if isvector(value) && numel(value) == T
            history.(fields{i}) = value(1:k);
        end
    end
    history.iterations = k;
    history.unit_accept_rate_U = mean(history.unit_accept_U);
    history.unit_accept_rate_V = mean(history.unit_accept_V);
    history.fallback_rate_U = mean(history.fallback_U);
    history.fallback_rate_V = mean(history.fallback_V);
    history.options = opts;

    F = (U + V) / 2;
    [~, labels] = max(F, [], 2);
end

function opts = local_defaults(opts)
    defaults.eta = 1;
    defaults.gamma0 = 1e-5;
    defaults.gamma_max = 1e8;
    defaults.eps_tau = 1e-12;
    defaults.max_iter = 3000;
    defaults.tol_step = 1e-4;
    defaults.tol_feas = 1e-4;
    defaults.beta = 0.5; 
    defaults.scaled_max_trials = 1; 
    defaults.fallback_max_trials = 1;
    defaults.theta = 0.95;
    defaults.armijo_c = 1e-8;
    defaults.numeric_tol = 1e-12;
    defaults.min_curvature = 1e-12;
    defaults.init_floor = 1e-12;
    defaults.seed = 1;
    defaults.verbose = false;
    defaults.print_every = 10;
    defaults.U0 = [];
    defaults.V0 = [];

    names = fieldnames(defaults);
    for i = 1:numel(names)
        name = names{i};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end
end
