%DEMO_SYNTHETIC  

rng(1);
n_per_cluster = 40; 
c = 3; 
d = 5; 
X = []; 
y = []; 
for j = 1:c 
    center = zeros(1, d); 
    center(j) = 4;
    X = [X; randn(n_per_cluster, d) + center]; %#ok<AGROW>
    y = [y; j * ones(n_per_cluster, 1)]; %#ok<AGROW>
end 

A = make_knn_affinity(X, 7);
opts.verbose = true;
opts.max_iter = 500;
opts.seed = 1;
opts.eta = 1;

[pred, U, V, Lambda, history] = dsd2f_psg_fallback(A, c, opts); %#ok<ASGLU>
fprintf('RI  = %.4f\n', rand_index_fast(y, pred));
fprintf('NMI = %.4f\n', nmi_score(y, pred));
fprintf('U unit-step acceptance = %.3f\n', history.unit_accept_rate_U);
fprintf('V unit-step acceptance = %.3f\n', history.unit_accept_rate_V); 
fprintf('U fallback rate = %.3f\n', history.fallback_rate_U); 
fprintf('V fallback rate = %.3f\n', history.fallback_rate_V);

plot_dsd2f_history(history, pwd, 'synthetic');
