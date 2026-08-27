% DEMO_DSD2F Minimal self-contained DSD2F example.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'));
addpath(fullfile(root,'utils'));

rng(1,'twister');
n = 120; r = 3;
gnd = repelem((1:r)', n/r);
A = sparse(n,n);
for i = 1:n
    same = find(gnd == gnd(i));
    same(same == i) = [];
    pick = same(randperm(numel(same), min(6,numel(same))));
    A(i,pick) = 1;
end
A = max(A,A');
Z0 = max(randn(n,r),0) + 1e-12;
[F,info] = dsd2f(A,r,Z0,dsd2f_options());

rng(1,'twister');
labels = kmeans(normalize_rows(F),r,'Start','plus','Replicates',10,'MaxIter',300);
fprintf('Demo RI: %.2f%%\n',100*rand_index(gnd,labels));
fprintf('Iterations: %d, final gamma: %.6g, balanced residual: %.3e\n', ...
    info.iterations,info.final_gamma,info.final_balanced_row_residual);
