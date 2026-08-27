function [U, V, info] = shared_balance_once(U, V, epsilon)
%SHARED_BALANCE_ONCE One shared diagonal correction without forming UV'.
%
% For S=(UV'+VU')/2, compute q=S*1 in O(nr), set
% d=(q+epsilon)^(-1/2), and replace U<-diag(d)U, V<-diag(d)V.

    n = size(U,1);
    one = ones(n,1);
    q = 0.5 * (U * (V' * one) + V * (U' * one));
    d = 1 ./ sqrt(max(q + epsilon, epsilon));
    U = U .* d;
    V = V .* d;

    if nargout > 2
        q_after = 0.5 * (U * (V' * one) + V * (U' * one));
        info = struct('residual', norm(q_after - one,2) / sqrt(n));
    end
end
