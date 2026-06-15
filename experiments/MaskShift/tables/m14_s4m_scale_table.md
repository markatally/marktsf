# M14 S4M scale-validation table

S4M is evaluated with 16 channels, 64 train windows, 48 test windows, and three seed offsets. This remains a MaskShift-protocol adaptation, not the full S4M benchmark protocol.

| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism mode | Kruskal p mean [95% CI] | Gate seeds |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Weather | S4M_official | 6.8% [-8.8, 22.4] | 0.109 [-0.134, 0.352] | mixed | 0.673 [0.000, 1.000] | 0/3 |
| Electricity | S4M_official | 2.5% [-17.9, 22.8] | 0.154 [-1.382, 1.690] | blackout | 0.989 [0.963, 1.000] | 0/3 |
