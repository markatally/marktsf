# M13 hierarchical bootstrap table

Bootstrap resamples lightweight variants and test windows. It is a hierarchy-aware uncertainty check for M1 aggregate claims, not a new model result.

| Dataset | eta^2 [95% CI] | Max abs delta [95% CI] | P(delta>0) | Loss-shift evidence | Worst tau [95% CI] | P(tau<=0.5) | Rank evidence |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Weather | 0.239 [0.088, 0.547] | 1.680 [0.514, 3.659] | 1.00 | SUPPORTED | 0.48 [-0.33, 1.00] | 0.46 | NOT_DECISIVE |
| Electricity | 0.320 [0.187, 0.538] | 1.446 [0.862, 1.943] | 1.00 | SUPPORTED | 0.37 [-0.33, 1.00] | 0.53 | NOT_DECISIVE |
| Traffic | 0.119 [0.007, 0.530] | 0.347 [-0.356, 1.101] | 0.83 | MIXED | 0.63 [0.00, 1.00] | 0.23 | NOT_DECISIVE |
| AirConvection | 0.163 [0.046, 0.501] | 12.269 [0.348, 33.052] | 1.00 | SUPPORTED | 0.84 [0.67, 1.00] | 0.02 | NOT_DECISIVE |
