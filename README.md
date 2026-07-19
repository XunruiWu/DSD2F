# DSD2F MATLAB Code

## 1. Correspondence to the Paper’s Formulas

The main function `dsd2f_psg_fallback.m` implements the following steps:

1. **Scaled trial for U**  
   \[
   [U-\alpha\nabla_U\ell\,(M_U)^{-1}]_+.
   \]
   The code first tries `alpha=1`. If the descent check fails, it reduces α at most 5 times.

2. **Projected‑gradient fallback for U**  
   \[
   [U-t_U\nabla_U\ell]_+,\qquad t_U=\theta/L_U.
   \]
   This step corresponds to the provable descent bound in the Proposition of the paper.

3. **Scaled trial and fallback for V**: handled in exactly the same way.

4. **KKT‑based multiplier update**  
   \[
   \Lambda=\operatorname{diag}\big([AV-UV^TV+\gamma(V-U)]U^T\big).
   \]
   The code computes this via row‑wise inner products, without forming the n×n matrix.

5. **Gamma protection**:
   - `tau >= 1` ensures that gamma never decreases;
   - `eps_tau` prevents division by too small a denominator;
   - `gamma_max` avoids numerical overflow.

6. **Stopping criteria**: both relative changes of U/V and the row‑sum residual are checked; the Lagrangian value under different Lambda and gamma is no longer compared.

## 2. Quick Test

Add the current directory to the MATLAB path and run:

matlab
demo_synthetic
