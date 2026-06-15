# M16 official TSLib full-coverage table

PatchTST and TimeXer are imported from pinned TSLib model classes. Weather/Electricity rows reuse M9; Traffic/AirConvection rows are new M16 coverage runs under the same MaskShift encoder-mask protocol.

| Dataset | Source | Official architecture classes | Max degradation | Worst tau | ANOVA p | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Weather | M9 | PatchTST_official, TimeXer_official | 123.7% | -1.000 | 6.66e-06 | PASS |
| Electricity | M9 | PatchTST_official, TimeXer_official | 87.8% | -1.000 | 0.00229 | PASS |
| Traffic | M16 | PatchTST_official, TimeXer_official | 35.4% | 1.000 | 0.0854 | MIXED/NEGATIVE |
| AirConvection | M16 | PatchTST_official, TimeXer_official | 46.1% | 1.000 | 0.000122 | MIXED/NEGATIVE |
