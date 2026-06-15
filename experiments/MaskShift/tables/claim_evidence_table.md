# Claim-to-evidence table

| ID | Claim | Evidence | Limit | Required wording |
| --- | --- | --- | --- | --- |
| C1 | Matched missing rate is not a robustness certificate. | M1, M3, M7, M9, M13 | Supported on Weather/Electricity; Traffic/AirConvection mixed; M13 supports loss-delta uncertainty but not universal rank instability. | State as evidence-backed benchmark finding, not universal theorem. |
| C2 | Mechanism shift can reverse model rankings. | M9 official-architecture adaptation; M1 ranks as supporting diagnostic | M9 Weather/Electricity worst tau=-1 over three seed offsets; M13 lightweight-rank bootstrap is not decisive. | Anchor rank reversal to M9/M10, not lightweight M1 alone. |
| C3 | The result is not only sensor retirement. | M8 non-retirement decomposition | Weather/Electricity pass without retirement. | Retirement remains a strong and obvious mechanism; keep decomposition visible. |
| C4 | Typed head is not a new method contribution. | M2/H3 | H3 fails; overall typed p=0.214. | Present as negative/diagnostic ablation. |
| C5 | Official modern architectures are affected. | M9/M10 | PatchTST/TimeXer official classes under custom MaskShift protocol. | Call it official-architecture adaptation, not full official benchmark reproduction. |
| C6 | Missing-aware architecture coverage is included but not decisive. | M11, M12, M14 | CTF_missing shows strong Weather sensitivity but weaker/non-significant Electricity sensitivity; S4M is negative/contrastive with 0/3 gate seeds under both reduced and larger-reduced protocols. | Report architecture-dependent evidence, not a win/loss claim. |
