# PRISM Paper-Ready Reproduction

Run from the repository root with the bundled/scientific Python environment.

```bash
PY=/Users/markguo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
$PY -m experiments.PRISM.router_viability
$PY -m experiments.PRISM.drift_beta_loop
$PY -m experiments.PRISM.ablations_significance
$PY -m experiments.PRISM.paper_ready
```

## Final Route

ETT-only empirical/pivot paper:

- M1b finance gate failed under the strict preregistered condition.
- M2 learned router failed on ETTm2 against Fixed-Share.
- M3/M4 retain dynamic beta as a small, statistically reliable contribution.
- Drift-triggered share-rate adaptation is rejected in the current form.

## M2 Router Viability

| Dataset | Fixed-Share | Descriptor Ridge | PRISM Router | Gate |
| --- | --- | --- | --- | --- |
| ETTh1 | 0.0476854 | 0.061286 | 0.0440427 | PASS |
| ETTm2 | 0.0609315 | 0.0998626 | 0.0736675 | FAIL |
| Weather | 0.00112011 | 0.00100759 | 0.000675396 | PASS |

## M3 Dynamic Beta / Drift Stress

| Dataset | Plain Stress | Loop Stress | Improvement | Beta IQR |
| --- | --- | --- | --- | --- |
| ETTh1 | 0.0437757 | 0.0437486 | 0.0618% | 0.262 |
| ETTm2 | 0.0613521 | 0.0613386 | 0.022% | 0.334 |
| Weather | 0.000989311 | 0.000987147 | 0.219% | 0.271 |

## M4 FDR Ablations

| Dataset | Plain FS | Full | Improvement | FDR |
| --- | --- | --- | --- | --- |
| ETTh1 | 0.0451876 | 0.0451628 | 0.0547% | PASS |
| ETTm2 | 0.0570956 | 0.0570818 | 0.0241% | PASS |
| Weather | 0.000985513 | 0.000983586 | 0.196% | PASS |

## Synthetic Identifiability

- State recovery accuracy: 0.966
- Best single loss: 0.6791
- Oracle loss: 0.4372
- Descriptor router loss: 0.4550
