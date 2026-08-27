DSD2F MATLAB Public Code Package (V4 Table 9/10)

This package contains only the DSD2F code itself and the scripts required to reproduce the DSD2F column in Tables 9/10 of this paper. It does not include third‑party methods such as DCD, LoRD, or B‑LoRD, nor does it provide download scripts for them.

Main files:

src/dsd2f.m – Numerical implementation of DSD2F used for V4 Tables 9/10.

src/dsd2f_options.m – Frozen V4 parameter settings.

src/shared_balance_once.m – One low‑rank feasibility correction after each complete U/V sweep.

src/final_symmetric_balance.m – High‑precision low‑rank balancing after termination.

RUN_DSD2F_TABLE9_10.m – One‑click entry for 10 datasets × 30 seeds.

reference/ – Frozen reference values for DSD2F reported in Tables 9/10.

Usage

After placing the prepared .mat data files as described in data/README.md, run from the MATLAB root directory:

matlab
RUN_DSD2F_TABLE9_10
For a quick test:

matlab
run(fullfile('tests','test_dsd2f_smoke.m'))
V4 Settings

r = c;

seeds = 1–30;

rng(seed, 'twister');

Z0 = max(randn(n, r), 0) + 1e-12, and set U0 = V0 = Z0;

eta = 50;

gamma0 = 1e-5, adaptively updated and clipped to [1e-4, 1e3];

maximum 3000 sweeps;

relative reconstruction objective tolerance 1e-6;

one shared low‑rank balancing correction after each complete sweep;

final balancing tolerance 1e-10, with at most 500 iterations;

final embeddings are row‑L2‑normalised, then clustered by MATLAB kmeans with 10 replicates and MaxIter = 300.

Solver time is measured only around the dsd2f call; it does not include data loading, common random initialisation, or the final k‑means clustering. The balancing operations performed inside dsd2f are included in the solver time.

Note: Tables 9/10 are based on an independent clean rerun (R2); for example, 20NEWS uses 18,846 samples and CANCER uses 190 samples. This package is intended solely for reproducing the DSD2F results in Tables 9/10 and not for recomputing the earlier Tables 3/4.
