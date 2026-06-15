# M11 official ChannelTokenFormer-missing adaptation table

ChannelTokenFormer_missing is imported from the official repository and evaluated under the MaskShift encoder-mask protocol. This is not the full CTF practical/irregular benchmark pipeline.

| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism | Gate seeds |
| --- | --- | --- | --- | --- | --- |
| Weather | ChannelTokenFormer_missing_official | 96.0% [39.4, 152.6] | 0.264 [0.003, 0.524] | volatility | 1/3 |
| Electricity | ChannelTokenFormer_missing_official | 32.3% [-10.4, 75.0] | 0.584 [-0.130, 1.297] | value_high | 0/3 |
