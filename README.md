# DSD2F MATLAB 代码

## 1. 与论文公式的对应关系

主函数 `dsd2f_psg_fallback.m` 实现以下步骤：

1. **U 的 scaled trial**
   \[
   [U-\alpha\nabla_U\ell\,(M_U)^{-1}]_+.
   \]
   程序默认先取 `alpha=1`。若下降检查不通过，最多缩小 5 次。

2. **U 的 projected-gradient fallback**
   \[
   [U-t_U\nabla_U\ell]_+,\qquad t_U=\theta/L_U.
   \]
   该步对应论文 Proposition 中的可证明下降界。

3. **V 的 scaled trial 与 fallback**：处理方式完全相同。

4. **KKT-based 乘子更新**
   \[
   \Lambda=\operatorname{diag}\big([AV-UV^TV+\gamma(V-U)]U^T\big).
   \]
   代码通过逐行内积计算，不形成 n×n 矩阵。

5. **gamma 保护**：
   - `tau >= 1`，确保 gamma 不减小；
   - `eps_tau` 防止分母过小；
   - `gamma_max` 防止数值溢出。

6. **停止条件**：同时检查 U/V 相对变化和 row-sum residual，不再比较不同 Lambda、gamma 下的 Lagrangian 值。

## 2. 快速测试

在 MATLAB 中把本目录加入路径，然后运行：

matlab
demo_synthetic

## 3. 单数据集重复实验

MAT 文件可包含：

- `A`：已构造好的 affinity matrix；或
- `X`：每行一个样本，程序据此构造 k-NN 图；
- 标签变量名支持 `gnd`、`labels`、`y`、`truth`。

## 4. r 与 k 敏感性实验

matlab
opts.runs = 10;
opts.k_values = [5, 7, 10, 15];
opts.rank_factors = [0.5, 0.75, 1, 1.25, 1.5];
results = run_sensitivity_experiments('CITESEER.mat', ...
    'results/CITESEER_sensitivity', opts);


输出 `sensitivity_summary.csv`，包含 RI、NMI、CPU time、unit-step 接受率和 fallback 率。

## 5. 正式重跑前需确认

- `eta` 应与最终论文设定一致；当前默认值为 1。
- 原论文中的 Tables 3--5 和 Figures 1--5 必须用这版代码重新生成。
- 大数据集建议使用稀疏 `A`，避免保存稠密 n×n affinity matrix。
- 当前环境没有 MATLAB/Octave，代码已进行静态检查，但正式实验前仍应先运行 `demo_synthetic.m`。

## 6. 梯度与下降界自检

matlab
validate_gradients

该脚本用有限差分检查 U/V 梯度，并验证 U-block fallback 的理论下降不等式。


## 逐行中文注释版说明

本目录中的每个 `.m` 文件均保留原算法语句，并对每一条非空 MATLAB 代码行添加中文行内说明。已有的中文注释保持不变。正式运行时无需删除这些注释。

建议依次运行：

matlab
validate_gradients
demo_synthetic

确认梯度误差、fallback 下降检验和合成数据演示均正常后，再运行正式数据集脚本。
