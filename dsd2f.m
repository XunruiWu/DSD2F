function [labels, U, V, Lambda, history] = dsd2f_psg_fallback(A, r, opts)  % 中文说明：定义 MATLAB 函数 “dsd2f_psg_fallback”并声明输入与输出。
%DSD2F_PSG_FALLBACK  DSD2F 的 projected scaled-gradient 实现。
%
% 该程序对应论文最新版中的算法：
%   1) 先尝试 projected scaled-gradient trial step；
%   2) 若有限次 trial 都不能满足下降条件，则切换到标准 projected-gradient fallback；
%   3) 用 KKT 关系给出的闭式公式更新乘子 Lambda；
%   4) 对 gamma 使用下界、分母和上界三重保护。
%
% 输入：
%   A    - n×n affinity matrix，可为稀疏矩阵
%   r    - 低秩因子列数
%   opts - 可选参数结构体，详见 local_defaults
%
% 输出：
%   labels  - 由 (U+V)/2 的逐行最大元素得到的聚类标签
%   U,V     - 最终两个非负因子
%   Lambda  - 最终乘子向量
%   history - 目标值、残差、步长接受率、fallback 次数等记录

    if nargin < 3  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        opts = struct();  % 中文说明：计算并更新保存算法或实验参数的结构体。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    opts = local_defaults(opts);  % 中文说明：补齐未由用户提供的算法参数。

    % ---------- 输入检查 ----------
    [n, m] = size(A);  % 中文说明：读取输入矩阵的行数和列数。
    if n ~= m  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        error('A 必须是方阵。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    if r < 1 || r ~= floor(r)  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        error('r 必须是正整数。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    if any(~isfinite(nonzeros(A)))  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        error('A 中包含 NaN 或 Inf。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    % 保留稀疏结构；负数会破坏 affinity 的含义，因此直接报错。
    if any(nonzeros(A) < 0)  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        error('A 应为非负 affinity matrix。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    one = ones(n, 1);  % 中文说明：创建全 1 数组并赋给 one。
    Ir = eye(r);  % 中文说明：计算并更新r 阶单位矩阵。
    rng(opts.seed, 'twister');  % 中文说明：设置随机数种子以保证实验可复现。

    % ---------- 初始化 ----------
    if isempty(opts.U0)  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        U = max(randn(n, r), 0);  % 中文说明：生成非负随机初值并赋给 U。
    else  % 中文说明：进入前述条件不成立时的分支。
        U = max(opts.U0, 0);  % 中文说明：计算并更新第一个非负因子矩阵。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    if isempty(opts.V0)  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        V = max(randn(n, r), 0);  % 中文说明：生成非负随机初值并赋给 V。
    else  % 中文说明：进入前述条件不成立时的分支。
        V = max(opts.V0, 0);  % 中文说明：计算并更新第二个非负因子矩阵。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    if ~isequal(size(U), [n, r]) || ~isequal(size(V), [n, r])  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
        error('U0 和 V0 的尺寸必须为 n×r。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    % 避免某一列恰好全零，导致初期曲率过小。
    U = U + opts.init_floor;  % 中文说明：计算并更新第一个非负因子矩阵。
    V = V + opts.init_floor;  % 中文说明：计算并更新第二个非负因子矩阵。
    Lambda = zeros(n, 1);  % 中文说明：创建全零数组并赋给 Lambda。
    gamma = opts.gamma0;  % 中文说明：计算并更新因子一致性参数。

    % ||A||_F^2 只需计算一次；对稀疏 A 也不会转成稠密矩阵。
    A_fro2 = full(sum(nonzeros(A).^2));  % 中文说明：把当前标量结果转换为普通数值并赋给 A_fro2。

    % ---------- 预分配历史变量 ----------
    T = opts.max_iter;  % 中文说明：计算并更新最大迭代次数。
    history.model_obj = nan(T, 1);  % 中文说明：预分配 history.model_obj，用于保存每次迭代的该项记录。
    history.aug_obj_before_dual = nan(T, 1);  % 中文说明：预分配 history.aug_obj_before_dual，用于保存每次迭代的该项记录。
    history.row_residual = nan(T, 1);  % 中文说明：预分配 history.row_residual，用于保存每次迭代的该项记录。
    history.factor_residual = nan(T, 1);  % 中文说明：预分配 history.factor_residual，用于保存每次迭代的该项记录。
    history.rel_U = nan(T, 1);  % 中文说明：预分配 history.rel_U，用于保存每次迭代的该项记录。
    history.rel_V = nan(T, 1);  % 中文说明：预分配 history.rel_V，用于保存每次迭代的该项记录。
    history.gamma = nan(T, 1);  % 中文说明：预分配 history.gamma，用于保存每次迭代的该项记录。
    history.alpha_U = nan(T, 1);  % 中文说明：预分配 history.alpha_U，用于保存每次迭代的该项记录。
    history.alpha_V = nan(T, 1);  % 中文说明：预分配 history.alpha_V，用于保存每次迭代的该项记录。
    history.unit_accept_U = false(T, 1);  % 中文说明：预分配 history.unit_accept_U，用于保存每次迭代的该项记录。
    history.unit_accept_V = false(T, 1);  % 中文说明：预分配 history.unit_accept_V，用于保存每次迭代的该项记录。
    history.fallback_U = false(T, 1);  % 中文说明：预分配 history.fallback_U，用于保存每次迭代的该项记录。
    history.fallback_V = false(T, 1);  % 中文说明：预分配 history.fallback_V，用于保存每次迭代的该项记录。
    history.time = nan(T, 1);  % 中文说明：预分配 history.time，用于保存每次迭代的该项记录。

    tic_total = tic;  % 中文说明：启动计时器以统计运行时间。

    for k = 1:T  % 中文说明：开始按给定索引范围执行循环。
        U_old = U;  % 中文说明：计算并更新当前迭代开始时的 U。
        V_old = V;  % 中文说明：计算并更新当前迭代开始时的 V。

        % 当前固定 Lambda 和 gamma 下的 augmented-Lagrangian 值。
        L_start = dsd2f_augmented_value(A, A_fro2, U, V, Lambda, gamma, opts.eta);  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L_start。

        %% =============================================================
        %  U-block：先尝试 scaled trial，失败后使用有理论下降保证的 fallback
        % ==============================================================
        s = V' * one;  % s = V^T 1，尺寸 r×1
        MU = V' * V + opts.eta * (s * s') + gamma * Ir;  % 中文说明：计算并更新U 块的右侧缩放矩阵。
        GU = A * V + opts.eta * (one * s') - Lambda * s' + gamma * V;  % 中文说明：计算并更新U 块梯度中的常数项。
        gradU = U * MU - GU;  % 中文说明：计算并更新增广拉格朗日函数对 U 的梯度。

        % L_U = ||M_U||_2。M_U 为对称正定小矩阵，可用 eig 精确计算。
        LU = max(eig((MU + MU') / 2));  % 中文说明：由小型对称矩阵的最大特征值计算 LU。
        LU = max(real(LU), opts.min_curvature);  % 中文说明：计算并更新U 块梯度的 Lipschitz 常数上界。

        alphaU = 1;  % 中文说明：计算并更新U 块 scaled trial 的当前步长。
        acceptedU = false;  % 中文说明：计算并更新记录 U 块 trial 是否被接受。
        U_trial = U;  % 中文说明：计算并更新保存已通过下降检验的 U 候选点。
        L_Utrial = L_start;  % 中文说明：计算并更新U 块更新后的增广拉格朗日函数值。

        % 最多测试 scaled_max_trials 个 scaled trial。
        for j = 1:opts.scaled_max_trials  % 中文说明：开始按给定索引范围执行循环。
            % gradU / MU 等价于 gradU * inv(MU)，但数值上更稳定。
            U_candidate = max(U - alphaU * (gradU / MU), 0);  % 中文说明：计算并更新当前 U 块候选点。
            L_candidate = dsd2f_augmented_value( ...  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L_candidate。
                A, A_fro2, U_candidate, V, Lambda, gamma, opts.eta);  % 中文说明：继续补充上一行表达式或函数调用的参数。
            dU2 = norm(U_candidate - U, 'fro')^2;  % 中文说明：计算所需范数或归一化量并赋给 dU2。

            if L_candidate <= L_start - opts.armijo_c * dU2 + opts.numeric_tol  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
                U_trial = U_candidate;  % 中文说明：计算并更新保存已通过下降检验的 U 候选点。
                L_Utrial = L_candidate;  % 中文说明：计算并更新U 块更新后的增广拉格朗日函数值。
                acceptedU = true;  % 中文说明：计算并更新记录 U 块 trial 是否被接受。
                break;  % 中文说明：满足条件后提前退出当前循环。
            end  % 中文说明：结束当前条件、循环或函数代码块。
            alphaU = alphaU * opts.beta;  % 中文说明：计算并更新U 块 scaled trial 的当前步长。
        end  % 中文说明：结束当前条件、循环或函数代码块。

        if acceptedU  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
            U = U_trial;  % 中文说明：计算并更新第一个非负因子矩阵。
            history.alpha_U(k) = alphaU;  % 中文说明：保存当前迭代的统计量。
            history.unit_accept_U(k) = abs(alphaU - 1) <= eps;  % 中文说明：保存当前迭代的统计量。
        else  % 中文说明：进入前述条件不成立时的分支。
            % 标准 projected-gradient fallback：t_U = theta / L_U < 1/L_U。
            tU = opts.theta / LU;  % 中文说明：计算并更新U 块 projected-gradient fallback 的步长。
            U = max(U - tU * gradU, 0);  % 中文说明：计算并更新第一个非负因子矩阵。
            L_Utrial = dsd2f_augmented_value( ...  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L_Utrial。
                A, A_fro2, U, V, Lambda, gamma, opts.eta);  % 中文说明：继续补充上一行表达式或函数调用的参数。

            % 理论上该步满足下降。下面只做数值安全检查；若浮点误差导致失败，继续减小 tU。
            fallback_ok = L_Utrial <= L_start + opts.numeric_tol;  % 中文说明：计算并更新记录 fallback 是否满足数值下降检验。
            bt = 0;  % 中文说明：计算并更新fallback 回溯次数。
            while ~fallback_ok && bt < opts.fallback_max_trials  % 中文说明：在条件成立期间重复执行回溯或迭代。
                tU = tU * opts.beta;  % 中文说明：计算并更新U 块 projected-gradient fallback 的步长。
                U = max(U_old - tU * gradU, 0);  % 中文说明：计算并更新第一个非负因子矩阵。
                L_Utrial = dsd2f_augmented_value( ...  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L_Utrial。
                    A, A_fro2, U, V, Lambda, gamma, opts.eta);  % 中文说明：继续补充上一行表达式或函数调用的参数。
                fallback_ok = L_Utrial <= L_start + opts.numeric_tol;  % 中文说明：计算并更新记录 fallback 是否满足数值下降检验。
                bt = bt + 1;  % 中文说明：计算并更新fallback 回溯次数。
            end  % 中文说明：结束当前条件、循环或函数代码块。
            if ~fallback_ok  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
                error('U-block fallback 未能产生下降，请检查数值尺度或参数。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
            end  % 中文说明：结束当前条件、循环或函数代码块。
            history.alpha_U(k) = tU;  % 中文说明：保存当前迭代的统计量。
            history.fallback_U(k) = true;  % 中文说明：保存当前迭代的统计量。
        end  % 中文说明：结束当前条件、循环或函数代码块。

        %% =============================================================
        %  V-block：同样先尝试 scaled trial，再使用 projected-gradient fallback
        % ==============================================================
        R = U' * U;  % 中文说明：计算并更新U 转置乘 U 的 Gram 矩阵。
        MV = (1 + opts.eta * n) * R + gamma * Ir;  % 中文说明：计算并更新V 块的右侧缩放矩阵。

        VR_minus_U = V * R - U;  % 中文说明：计算并更新V 乘 R 与 U 的差。
        % 下面两项只用向量和，不显式生成 1*1^T：
        %   1 Lambda^T U  = repmat(Lambda^T U, n, 1)
        %   1 1^T(VR-U)  = repmat(sum(VR-U,1), n, 1)
        gradV = V * R - A' * U ...  % 中文说明：计算并更新增广拉格朗日函数对 V 的梯度。
              + one * (Lambda' * U) ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
              + opts.eta * one * sum(VR_minus_U, 1) ...  % 中文说明：继续补充上一行表达式或函数调用的参数。
              + gamma * (V - U);  % 中文说明：继续补充上一行表达式或函数调用的参数。

        LV = (1 + opts.eta * n) * max(eig((R + R') / 2)) + gamma;  % 中文说明：由小型对称矩阵的最大特征值计算 LV。
        LV = max(real(LV), opts.min_curvature);  % 中文说明：计算并更新V 块梯度的 Lipschitz 常数上界。

        L_after_U = L_Utrial;  % 中文说明：计算并更新完成 U 更新后的函数值。
        alphaV = 1;  % 中文说明：计算并更新V 块 scaled trial 的当前步长。
        acceptedV = false;  % 中文说明：计算并更新记录 V 块 trial 是否被接受。
        V_trial = V;  % 中文说明：计算并更新保存已通过下降检验的 V 候选点。
        L_Vtrial = L_after_U;  % 中文说明：计算并更新V 块更新后的增广拉格朗日函数值。

        for j = 1:opts.scaled_max_trials  % 中文说明：开始按给定索引范围执行循环。
            V_candidate = max(V - alphaV * (gradV / MV), 0);  % 中文说明：计算并更新当前 V 块候选点。
            L_candidate = dsd2f_augmented_value( ...  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L_candidate。
                A, A_fro2, U, V_candidate, Lambda, gamma, opts.eta);  % 中文说明：继续补充上一行表达式或函数调用的参数。
            dV2 = norm(V_candidate - V, 'fro')^2;  % 中文说明：计算所需范数或归一化量并赋给 dV2。

            if L_candidate <= L_after_U - opts.armijo_c * dV2 + opts.numeric_tol  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
                V_trial = V_candidate;  % 中文说明：计算并更新保存已通过下降检验的 V 候选点。
                L_Vtrial = L_candidate;  % 中文说明：计算并更新V 块更新后的增广拉格朗日函数值。
                acceptedV = true;  % 中文说明：计算并更新记录 V 块 trial 是否被接受。
                break;  % 中文说明：满足条件后提前退出当前循环。
            end  % 中文说明：结束当前条件、循环或函数代码块。
            alphaV = alphaV * opts.beta;  % 中文说明：计算并更新V 块 scaled trial 的当前步长。
        end  % 中文说明：结束当前条件、循环或函数代码块。

        if acceptedV  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
            V = V_trial;  % 中文说明：计算并更新第二个非负因子矩阵。
            history.alpha_V(k) = alphaV;  % 中文说明：保存当前迭代的统计量。
            history.unit_accept_V(k) = abs(alphaV - 1) <= eps;  % 中文说明：保存当前迭代的统计量。
        else  % 中文说明：进入前述条件不成立时的分支。
            tV = opts.theta / LV;  % 中文说明：计算并更新V 块 fallback 的步长。
            V = max(V - tV * gradV, 0);  % 中文说明：计算并更新第二个非负因子矩阵。
            L_Vtrial = dsd2f_augmented_value( ...  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L_Vtrial。
                A, A_fro2, U, V, Lambda, gamma, opts.eta);  % 中文说明：继续补充上一行表达式或函数调用的参数。

            fallback_ok = L_Vtrial <= L_after_U + opts.numeric_tol;  % 中文说明：计算并更新记录 fallback 是否满足数值下降检验。
            bt = 0;  % 中文说明：计算并更新fallback 回溯次数。
            while ~fallback_ok && bt < opts.fallback_max_trials  % 中文说明：在条件成立期间重复执行回溯或迭代。
                tV = tV * opts.beta;  % 中文说明：计算并更新V 块 fallback 的步长。
                V = max(V_old - tV * gradV, 0);  % 中文说明：计算并更新第二个非负因子矩阵。
                L_Vtrial = dsd2f_augmented_value( ...  % 中文说明：调用辅助函数计算增广拉格朗日函数值并赋给 L_Vtrial。
                    A, A_fro2, U, V, Lambda, gamma, opts.eta);  % 中文说明：继续补充上一行表达式或函数调用的参数。
                fallback_ok = L_Vtrial <= L_after_U + opts.numeric_tol;  % 中文说明：计算并更新记录 fallback 是否满足数值下降检验。
                bt = bt + 1;  % 中文说明：计算并更新fallback 回溯次数。
            end  % 中文说明：结束当前条件、循环或函数代码块。
            if ~fallback_ok  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
                error('V-block fallback 未能产生下降，请检查数值尺度或参数。');  % 中文说明：检测到非法输入或数值失败时终止程序并给出提示。
            end  % 中文说明：结束当前条件、循环或函数代码块。
            history.alpha_V(k) = tV;  % 中文说明：保存当前迭代的统计量。
            history.fallback_V(k) = true;  % 中文说明：保存当前迭代的统计量。
        end  % 中文说明：结束当前条件、循环或函数代码块。

        %% =============================================================
        %  KKT-based 乘子闭式更新
        % ==============================================================
        AV = A * V;  % 中文说明：计算并更新亲和矩阵 A 与 V 的乘积。
        VtV = V' * V;  % 中文说明：计算并更新V 的 Gram 矩阵。
        multiplier_core = AV - U * VtV + gamma * (V - U);  % 中文说明：计算并更新闭式乘子公式中与 U 做逐行内积的矩阵。

        % diag(multiplier_core * U^T) 等于逐行内积，避免形成 n×n 矩阵。
        Lambda_new = sum(multiplier_core .* U, 2);  % 中文说明：计算并更新根据 KKT 闭式关系得到的新乘子。

        %% =============================================================
        %  gamma 自适应更新：确保非递减，并加分母保护和上界
        % ==============================================================
        innerUV = sum(U(:) .* V(:));  % 中文说明：计算并更新U 与 V 的 Frobenius 内积。
        ratio = (norm(U, 'fro')^2 + norm(V, 'fro')^2) / ...  % 中文说明：计算所需范数或归一化量并赋给 ratio。
                (2 * max(abs(innerUV), opts.eps_tau));  % 中文说明：继续补充上一行表达式或函数调用的参数。
        tau = max(1, ratio);  % 中文说明：计算并更新不小于 1 的 gamma 放大系数。
        gamma_new = min(opts.gamma_max, gamma * tau);  % 中文说明：计算并更新加上上界后的新 gamma。

        %% =============================================================
        %  记录与停止判据
        % ==============================================================
        relU = norm(U - U_old, 'fro') / max(1, norm(U_old, 'fro'));  % 中文说明：计算所需范数或归一化量并赋给 relU。
        relV = norm(V - V_old, 'fro') / max(1, norm(V_old, 'fro'));  % 中文说明：计算所需范数或归一化量并赋给 relV。
        row_res = U * (V' * one) - one;  % 中文说明：计算并更新当前行和约束残差向量。
        feas = norm(row_res, 2) / sqrt(n);  % 中文说明：计算所需范数或归一化量并赋给 feas。
        factor_res = norm(U - V, 'fro') / max(1, norm(U, 'fro') + norm(V, 'fro'));  % 中文说明：计算所需范数或归一化量并赋给 factor_res。

        history.model_obj(k) = dsd2f_model_value(A, A_fro2, U, V, gamma);  % 中文说明：保存当前迭代的统计量。
        % 该值使用更新 Lambda/gamma 前的固定子问题参数，和论文中的下降命题对应。
        history.aug_obj_before_dual(k) = L_Vtrial;  % 中文说明：保存当前迭代的统计量。
        history.row_residual(k) = feas;  % 中文说明：保存当前迭代的统计量。
        history.factor_residual(k) = factor_res;  % 中文说明：保存当前迭代的统计量。
        history.rel_U(k) = relU;  % 中文说明：保存当前迭代的统计量。
        history.rel_V(k) = relV;  % 中文说明：保存当前迭代的统计量。
        history.gamma(k) = gamma_new;  % 中文说明：保存当前迭代的统计量。
        history.time(k) = toc(tic_total);  % 中文说明：保存当前迭代的统计量。

        Lambda = Lambda_new;  % 中文说明：计算并更新行和约束对应的乘子向量。
        gamma = gamma_new;  % 中文说明：计算并更新因子一致性参数。

        if opts.verbose && (k == 1 || mod(k, opts.print_every) == 0)  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
            fprintf(['iter=%4d  model=%.6e  feas=%.3e  relU=%.3e  relV=%.3e  ' ...  % 中文说明：把当前实验指标或迭代信息输出到命令窗口。
                     'gamma=%.3e  fbU=%d  fbV=%d\n'], ...  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
                k, history.model_obj(k), feas, relU, relV, gamma, ...  % 中文说明：执行本行所示的 MATLAB 运算或函数调用。
                history.fallback_U(k), history.fallback_V(k));  % 中文说明：继续补充上一行表达式或函数调用的参数。
        end  % 中文说明：结束当前条件、循环或函数代码块。

        if max(relU, relV) <= opts.tol_step && feas <= opts.tol_feas  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
            break;  % 中文说明：满足条件后提前退出当前循环。
        end  % 中文说明：结束当前条件、循环或函数代码块。
    end  % 中文说明：结束当前条件、循环或函数代码块。

    % 截断未使用的预分配空间。
    fields = fieldnames(history);  % 中文说明：读取 history 结构体的全部字段名。
    for i = 1:numel(fields)  % 中文说明：开始按给定索引范围执行循环。
        value = history.(fields{i});  % 中文说明：计算并更新当前函数返回的标量值。
        if isvector(value) && numel(value) == T  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
            history.(fields{i}) = value(1:k);  % 中文说明：保存当前迭代的统计量。
        end  % 中文说明：结束当前条件、循环或函数代码块。
    end  % 中文说明：结束当前条件、循环或函数代码块。
    history.iterations = k;  % 中文说明：把当前的 iterations 指标写入 history 结构体。
    history.unit_accept_rate_U = mean(history.unit_accept_U);  % 中文说明：把当前的 unit_accept_rate_U 指标写入 history 结构体。
    history.unit_accept_rate_V = mean(history.unit_accept_V);  % 中文说明：把当前的 unit_accept_rate_V 指标写入 history 结构体。
    history.fallback_rate_U = mean(history.fallback_U);  % 中文说明：把当前的 fallback_rate_U 指标写入 history 结构体。
    history.fallback_rate_V = mean(history.fallback_V);  % 中文说明：把当前的 fallback_rate_V 指标写入 history 结构体。
    history.options = opts;  % 中文说明：把当前的 options 指标写入 history 结构体。

    F = (U + V) / 2;  % 中文说明：计算右侧表达式并把结果赋给 F。
    [~, labels] = max(F, [], 2);  % 中文说明：按每行最大因子分量生成最终聚类标签。
end  % 中文说明：结束当前条件、循环或函数代码块。

function opts = local_defaults(opts)  % 中文说明：定义 MATLAB 函数 “local_defaults”并声明输入与输出。
%LOCAL_DEFAULTS  填充缺省参数；用户传入的字段优先。
    defaults.eta = 1;  % 中文说明：设置参数 eta 的缺省值。
    defaults.gamma0 = 1e-5;  % 中文说明：设置参数 gamma0 的缺省值。
    defaults.gamma_max = 1e8;  % 中文说明：设置参数 gamma_max 的缺省值。
    defaults.eps_tau = 1e-12;  % 中文说明：设置参数 eps_tau 的缺省值。
    defaults.max_iter = 3000;  % 中文说明：设置参数 max_iter 的缺省值。
    defaults.tol_step = 1e-4;  % 中文说明：设置参数 tol_step 的缺省值。
    defaults.tol_feas = 1e-4;  % 中文说明：设置参数 tol_feas 的缺省值。
    defaults.beta = 0.5;  % 中文说明：设置参数 beta 的缺省值。
    defaults.scaled_max_trials = 5;  % 中文说明：设置参数 scaled_max_trials 的缺省值。
    defaults.fallback_max_trials = 20;  % 中文说明：设置参数 fallback_max_trials 的缺省值。
    defaults.theta = 0.95;  % 中文说明：设置参数 theta 的缺省值。
    defaults.armijo_c = 1e-8;  % 中文说明：设置参数 armijo_c 的缺省值。
    defaults.numeric_tol = 1e-12;  % 中文说明：设置参数 numeric_tol 的缺省值。
    defaults.min_curvature = 1e-12;  % 中文说明：设置参数 min_curvature 的缺省值。
    defaults.init_floor = 1e-12;  % 中文说明：设置参数 init_floor 的缺省值。
    defaults.seed = 1;  % 中文说明：设置参数 seed 的缺省值。
    defaults.verbose = false;  % 中文说明：设置参数 verbose 的缺省值。
    defaults.print_every = 10;  % 中文说明：设置参数 print_every 的缺省值。
    defaults.U0 = [];  % 中文说明：设置参数 U0 的缺省值。
    defaults.V0 = [];  % 中文说明：设置参数 V0 的缺省值。

    names = fieldnames(defaults);  % 中文说明：读取 defaults 结构体的全部缺省参数名。
    for i = 1:numel(names)  % 中文说明：开始按给定索引范围执行循环。
        name = names{i};  % 中文说明：计算并更新当前处理的字段名。
        if ~isfield(opts, name) || isempty(opts.(name))  % 中文说明：判断当前条件是否成立，并在成立时进入该分支。
            opts.(name) = defaults.(name);  % 中文说明：设置当前实验或算法参数。
        end  % 中文说明：结束当前条件、循环或函数代码块。
    end  % 中文说明：结束当前条件、循环或函数代码块。
end  % 中文说明：结束当前条件、循环或函数代码块。
