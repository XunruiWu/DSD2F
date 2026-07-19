rng(2);
n = 12;
r = 4;
A = rand(n);
A = (A + A') / 2;
U = max(randn(n, r), 0) + 1e-3;
V = max(randn(n, r), 0) + 1e-3;
Lambda = randn(n, 1);
eta = 1.3;
gamma = 0.2;
one = ones(n, 1);
Ir = eye(r);
A_fro2 = norm(A, 'fro')^2;

s = V' * one;
MU = V' * V + eta * (s * s') + gamma * Ir;
GU = A * V + eta * one * s' - Lambda * s' + gamma * V;
gU = U * MU - GU;

R = U' * U;
gV = V * R - A' * U + one * (Lambda' * U) ...
   + eta * one * sum(V * R - U, 1) + gamma * (V - U);

h = 1e-6;
i = 3;
j = 2;

Up = U; Um = U;
Up(i,j) = Up(i,j) + h;
Um(i,j) = Um(i,j) - h;
fdU = (dsd2f_augmented_value(A, A_fro2, Up, V, Lambda, gamma, eta) ...
     - dsd2f_augmented_value(A, A_fro2, Um, V, Lambda, gamma, eta)) / (2*h);

Vp = V; Vm = V;
Vp(i,j) = Vp(i,j) + h;
Vm(i,j) = Vm(i,j) - h;
fdV = (dsd2f_augmented_value(A, A_fro2, U, Vp, Lambda, gamma, eta) ...
     - dsd2f_augmented_value(A, A_fro2, U, Vm, Lambda, gamma, eta)) / (2*h);

fprintf('U gradient: analytic=% .8e, finite-diff=% .8e, error=%.3e\n', ...
    gU(i,j), fdU, abs(gU(i,j)-fdU));
fprintf('V gradient: analytic=% .8e, finite-diff=% .8e, error=%.3e\n', ...
    gV(i,j), fdV, abs(gV(i,j)-fdV));

LU = max(eig((MU + MU') / 2));
theta = 0.95;
tU = theta / LU;
PU = max(U - tU * gU, 0);
L0 = dsd2f_augmented_value(A, A_fro2, U, V, Lambda, gamma, eta);
L1 = dsd2f_augmented_value(A, A_fro2, PU, V, Lambda, gamma, eta);
rhs = L0 - (1-theta)/(2*tU) * norm(PU-U, 'fro')^2;
fprintf('U fallback bound: L(new)=%.8e, RHS=%.8e, satisfied=%d\n', ...
    L1, rhs, L1 <= rhs + 1e-10);
