# MaskShift Dataset and Mechanism Cards

These cards are the reviewer-facing map from each dataset and mechanism to the exact claim scope. They are generated from M0, M3, M8, and M16 summaries.

## Dataset Cards

| Dataset | Rows | Channels | Natural missing | Max rate error | M1 eta^2 | M1 gate | M8 non-ret gate | M16 max degradation | M16 worst tau | Evidence role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weather | 7000 | 21 | 0.00% | 0.21% | 0.614 | PASS | PASS | 123.7% | -1.000 | positive core evidence |
| Electricity | 7000 | 24 | 0.00% | 0.19% | 0.777 | PASS | PASS | 87.8% | -1.000 | positive core evidence |
| Traffic | 7000 | 24 | 0.00% | 0.19% | 0.012 | MIXED | MIXED | 35.4% | 1.000 | boundary/mixed evidence |
| AirConvection | 7000 | 8 | 20.08% | 0.39% | 0.120 | MIXED | MIXED | 46.1% | 1.000 | boundary/mixed evidence |

## Mechanism Cards

### mcar

- Definition: Independent random deletion at the target missing rate.
- Deployment read: Neutral packet loss or randomly sampled telemetry dropout.
- Scope guard: Control mechanism only; not an operational cause model.

### block

- Definition: Contiguous channel-level missing spans with matched final rate.
- Deployment read: Short sensor logging gaps or temporary channel outages.
- Scope guard: Controls contiguity but not value dependence.

### value_high

- Definition: Missing probability increases at high normalized values, then rate matching is enforced.
- Deployment read: Storm, peak-load, congestion, or saturation-related missingness.
- Scope guard: Synthetic proxy for value-dependent MNAR; not a causal claim.

### volatility

- Definition: Missing probability increases with large local changes, then rate matching is enforced.
- Deployment read: Failure during unstable regimes, transitions, or high-frequency variation.
- Scope guard: Measures sensitivity to change-linked masks, not event detection quality.

### blackout

- Definition: Contiguous time blocks remove many channels simultaneously with matched final rate.
- Deployment read: Power/network outage, site-level telemetry loss, or shared infrastructure failure.
- Scope guard: Does not model outage recovery dynamics beyond the encoder window.

### retirement

- Definition: Selected channels disappear late in the sequence with matched final rate.
- Deployment read: Permanent sensor retirement, meter replacement, or channel decommissioning.
- Scope guard: Reported separately so the paper is not driven only by an obvious failure case.

## Claim Boundary

Weather and Electricity carry the strongest positive evidence. Traffic and AirConvection are kept as boundary evidence, not hidden negative cases. The paper should claim that missingness mechanism is a first-class benchmark factor whose empirical strength is dataset- and architecture-dependent.
