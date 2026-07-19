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
