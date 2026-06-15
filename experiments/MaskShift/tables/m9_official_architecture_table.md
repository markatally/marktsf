# M9 official-architecture adaptation table

PatchTST and TimeXer are imported from pinned TSLib model classes; the MaskShift loop is a custom encoder-mask protocol, not the full official benchmark protocol.

| Dataset | Official architecture classes | Max degradation mean [95% CI] | Worst tau mean [95% CI] | Gate seeds |
| --- | --- | --- | --- | --- |
| Weather | PatchTST_official, TimeXer_official | 135.8% [8.6, 262.9] | -1.00 [-1.00, -1.00] | 3/3 |
| Electricity | PatchTST_official, TimeXer_official | 128.6% [98.1, 159.1] | -1.00 [-1.00, -1.00] | 3/3 |
