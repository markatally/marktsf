# M7 corrected robustness table

The original relative degradation ratio is retained only as a diagnostic because small MCAR denominators can explode. The submission reports absolute delta, log ratio, and symmetric relative delta.

| Missing rate | eta^2 mean [95% CI] | Max abs delta [95% CI] | Max log ratio [95% CI] | Max symmetric delta [95% CI] | Denom unstable |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 0.194 [0.000, 0.456] | 0.540 [0.225, 0.855] | 0.66 [0.01, 1.30] | 0.62 [0.06, 1.17] | 0/4 |
| 0.20 | 0.286 [0.000, 0.657] | 10.713 [-22.257, 43.683] | 1.33 [-1.55, 4.22] | 0.80 [-0.45, 2.05] | 0/4 |
| 0.35 | 0.423 [0.000, 0.881] | 2.577 [-3.801, 8.954] | 1.27 [-0.40, 2.94] | 0.98 [-0.16, 2.11] | 0/4 |
| 0.50 | 0.472 [0.060, 0.884] | 0.854 [0.452, 1.256] | 0.92 [0.01, 1.83] | 0.82 [0.13, 1.51] | 0/4 |
