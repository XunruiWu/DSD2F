%VALIDATE_GRADIENTS  用有限差分检查 U/V 梯度和 fallback 下降界。
% 建议在正式实验前运行一次；相对误差通常应在 1e-5 左右或更小。

rng(2);  % 中文说明：设置随机数种子以保证实验可复现。
n = 12;  % 中文说明：计算并更新样本数。
r = 4;  % 中文说明：计算并更新低秩因子列数。
A = rand(n);  % 中文说明：计算并更新亲和矩阵。
A = (A + A') / 2;  % 中文说明：计算并更新亲和矩阵。
U = max(randn(n, r), 0) + 1e-3;  % 中文说明：生成非负随机初值并赋给 U。
V = max(randn(n, r), 0) + 1e-3;  % 中文说明：生成非负随机初值并赋给 V。
Lambda = randn(n, 1);  % 中文说明：计算并更新行和约束对应的乘子向量。
eta = 1.3;  % 中文说明：计算右侧表达式并把结果赋给 eta。
gamma = 0.2;  % 中文说明：计算并更新因子一致性参数。
one = ones(n, 1);  % 中文说明：创建全 1 数组并赋给 one。
Ir = eye(r);  % 中文说明：计算并更新r 阶单位矩阵。
A_fro2 = norm(A, 'fro')^2;  % 中文说明：计算所需范数或归一化量并赋给 A_fro2。

s = V' * one;  % 中文说明：计算并更新向量 V 转置乘全 1 向量。
MU = V' * V + eta * (s * s') + gamma * Ir;  % 中文说明：计算并更新U 块的右侧缩放矩阵。
GU = A * V + eta * one * s' - Lambda * s' + gamma * V;  % 中文说明：计算并更新U 块梯度中的常数项。
gU = U * MU - GU;  % 中文说明：计算并更新解析得到的 U 梯度。

R = U' * U;  % 中文说明：计算并更新U 转置乘 U 的 Gram 矩阵。
gV = V * R - A' * U + one * (Lambda' * U) ...  % 中文说明：计算并更新解析得到的 V 梯度。
   + eta * one * sum(V * R - U, 1) + gamma * (V - U);  % 中文说明：继续补充上一行表达式或函数调用的参数。

h = 1e-6;  % 中文说明：计算并更新有限差分步长。
i = 3;  % 中文说明：计算并更新当前索引。
j = 2;  % 中文说明：计算并更新当前循环索引。

Up = U; Um = U;  % 中文说明：计算右侧表达式并把结果赋给 Up。
Up(i,j) = Up(i,j) + h;  % 中文说明：对指定元素加入正向或负向有限差分扰动。
Um(i,j) = Um(i,j) - h;  % 中文说明：对指定元素加入正向或负向有限差分扰动。
fdU = (dsd2f_augmented_value(A, A_fro2, Up, V, Lambda, gamma, eta) ...  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 fdU。
     - dsd2f_augmented_value(A, A_fro2, Um, V, Lambda, gamma, eta)) / (2*h);  % 中文说明：继续补充上一行表达式或函数调用的参数。

Vp = V; Vm = V;  % 中文说明：计算右侧表达式并把结果赋给 Vp。
Vp(i,j) = Vp(i,j) + h;  % 中文说明：对指定元素加入正向或负向有限差分扰动。
Vm(i,j) = Vm(i,j) - h;  % 中文说明：对指定元素加入正向或负向有限差分扰动。
fdV = (dsd2f_augmented_value(A, A_fro2, U, Vp, Lambda, gamma, eta) ...  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 fdV。
     - dsd2f_augmented_value(A, A_fro2, U, Vm, Lambda, gamma, eta)) / (2*h);  % 中文说明：继续补充上一行表达式或函数调用的参数。

fprintf('U gradient: analytic=% .8e, finite-diff=% .8e, error=%.3e\n', ...  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
    gU(i,j), fdU, abs(gU(i,j)-fdU));  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
fprintf('V gradient: analytic=% .8e, finite-diff=% .8e, error=%.3e\n', ...  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
    gV(i,j), fdV, abs(gV(i,j)-fdV));  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。

% 检查 U fallback 的理论下降不等式。
LU = max(eig((MU + MU') / 2));  % 中文说明：由小型对称矩阵的最大特征值计算 LU。
theta = 0.95;  % 中文说明：计算右侧表达式并把结果赋给 theta。
tU = theta / LU;  % 中文说明：计算并更新U 块 projected-gradient fallback 的步长。
PU = max(U - tU * gU, 0);  % 中文说明：计算并更新U 块 fallback 得到的投影点。
L0 = dsd2f_augmented_value(A, A_fro2, U, V, Lambda, gamma, eta);  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L0。
L1 = dsd2f_augmented_value(A, A_fro2, PU, V, Lambda, gamma, eta);  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L1。
rhs = L0 - (1-theta)/(2*tU) * norm(PU-U, 'fro')^2;  % 中文说明：计算所需范数或归一化量并赋给 rhs。
fprintf('U fallback bound: L(new)=%.8e, RHS=%.8e, satisfied=%d\n', ...  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
    L1, rhs, L1 <= rhs + 1e-10);  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
