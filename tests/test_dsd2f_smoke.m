% TEST_DSD2F_SMOKE Basic structural and numerical checks.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'));

rng(7,'twister');
n = 60; r = 4;
R = sprand(n,n,0.08);
A = max(R,R');
A = spones(A);
A = A - spdiags(diag(A),0,n,n);
Z0 = max(randn(n,r),0) + 1e-12;
opts = dsd2f_options();
opts.max_iter = 30;
[F,info] = dsd2f(A,r,Z0,opts);
assert(all(size(F) == [n,r]));
assert(all(isfinite(F(:))));
assert(all(F(:) >= 0));
assert(isfinite(info.final_gamma));
assert(info.final_gamma >= opts.gamma_min && info.final_gamma <= opts.gamma_max);
assert(isfinite(info.final_balanced_row_residual));
fprintf('DSD2F smoke test passed. final balanced residual = %.3e\n', ...
    info.final_balanced_row_residual);
