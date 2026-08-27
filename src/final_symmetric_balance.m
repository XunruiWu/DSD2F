function [d, info] = final_symmetric_balance(U, V, epsilon, tolerance, max_iter)
%FINAL_SYMMETRIC_BALANCE High-accuracy low-rank symmetric balancing.
%
% For S=(UV'+VU')/2, solve d.*(S*d)=1 by the positive fixed-point update
%       d <- sqrt( d ./ (S*d + epsilon) )
% using only O(nr) factor-vector products.

    n = size(U,1);
    one = ones(n,1);
    d = ones(n,1);
    residual = inf;

    for k = 1:max_iter
        Sd = 0.5 * (U * (V' * d) + V * (U' * d));
        residual = norm(d .* Sd - one, 2) / sqrt(n);
        if residual <= tolerance
            break;
        end
        d = sqrt(d ./ max(Sd + epsilon, epsilon));
    end

    % Recompute the residual for the returned scaling.
    Sd = 0.5 * (U * (V' * d) + V * (U' * d));
    residual = norm(d .* Sd - one, 2) / sqrt(n);

    info = struct();
    info.iterations = k;
    info.residual = residual;
end
