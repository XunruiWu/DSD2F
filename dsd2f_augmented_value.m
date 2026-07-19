function value = dsd2f_augmented_value(A, A_fro2, U, V, Lambda, gamma, eta)  % 中文说明：定义 MATLAB 函数 “dsd2f_augmented_value”并声明输入与输出。
%DSD2F_AUGMENTED_VALUE  计算固定 Lambda、gamma 下的 augmented Lagrangian。
% 不显式形成 UV^T，适用于较大的稀疏 affinity matrix。

    n = size(U, 1);  % 中文说明：读取矩阵尺寸并赋给 n。
    one = ones(n, 1);  % 中文说明：创建全 1 数组并赋给 one。

    UtU = U' * U;  % 中文说明：计算并更新U 的 Gram 矩阵。
    VtV = V' * V;  % 中文说明：计算并更新V 的 Gram 矩阵。
    AV = A * V;  % 中文说明：计算并更新亲和矩阵 A 与 V 的乘积。

    % ||UV^T-A||_F^2 = ||A||_F^2 + tr(U^TU V^TV) - 2 tr(U^T A V)
    fit2 = A_fro2 + trace(UtU * VtV) - 2 * sum(sum(U .* AV));  % 中文说明：计算并更新拟合误差平方。
    fit2 = max(real(fit2), 0);  % 仅消除浮点舍入导致的极小负数

    residual = U * (V' * one) - one;  % 中文说明：计算并更新行和约束残差。
    consistency = norm(U - V, 'fro')^2;  % 中文说明：计算所需范数或归一化量并赋给 consistency。

    value = 0.5 * fit2 ...  % 中文说明：计算并更新当前函数返回的标量值。
          + Lambda' * residual ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
          + 0.5 * eta * (residual' * residual) ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
          + 0.5 * gamma * consistency;  % 中文说明：继续补充上一行表达式或函数调用的参数。
    value = real(value);  % 中文说明：计算并更新当前函数返回的标量值。
end  % 中文说明：结束当前条件、循环或函数代码块。
