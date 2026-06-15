# M12 official S4M adaptation table

S4M is imported from the official repository and evaluated under the MaskShift encoder-mask protocol with reduced channels/samples and three seed offsets. This is not the full S4M benchmark reproduction.

| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism mode | Kruskal p mean [95% CI] | Gate seeds |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Weather | S4M_official | 5.7% [-44.9, 56.3] | 0.071 [-0.620, 0.762] | mixed | 0.509 [0.081, 0.938] | 0/3 |
| Electricity | S4M_official | 6.2% [-0.1, 12.5] | 0.312 [0.093, 0.530] | mixed | 0.933 [0.727, 1.000] | 0/3 |
