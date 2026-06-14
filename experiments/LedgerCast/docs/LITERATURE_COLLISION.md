# LedgerCast Literature Collision Log

## Search Queries

- "hierarchical time series forecasting neural reconciliation deep learning ICML ICLR NeurIPS"
- "learning optimal projection forecast reconciliation hierarchical time series ICML"
- "coherent probabilistic forecasting hierarchical time series neural"
- "noisy hierarchy forecast reconciliation revisions missing bottom nodes"
- "forecast reconciliation robust contaminated hierarchy"

## Closest Works and Claim-Level Overlap

| Work | Link | Claim overlap | Collision severity | Resolution |
|---|---|---:|---:|---|
| Optimal Forecast Reconciliation (MinT) | https://ideas.repec.org/a/taf/jnlasa/v114y2019i526p804-819.html | Classical coherent projection | High | Mandatory baseline |
| End-to-End Coherent Probabilistic Forecasts | https://proceedings.mlr.press/v139/rangapuram21a.html | Neural coherent probabilistic forecasts | High | Baseline; exact hierarchy |
| HINT / hierarchical mixture networks | https://arxiv.org/abs/2305.07089 | Neural hierarchical probabilistic forecasts | High | Baseline |
| Coherent Probabilistic Temporal Hierarchies | https://proceedings.mlr.press/v206/rangapuram23a.html | Temporal aggregation coherence | Medium-high | Baseline for temporal hierarchies |
| Graph-based TS clustering/reconciliation | https://arxiv.org/abs/2305.19183 | Graph/hierarchy learning and reconciliation | Medium-high | Related but not noisy-ledger reliability |
| Learning Optimal Projection | https://research.ibm.com/publications/learning-optimal-projection-for-forecast-reconciliation-of-hierarchical-time-series | Learnable oblique projection | High | Strongest method baseline |
| NeuralReconciler | https://dl.acm.org/doi/10.1145/3616855.3635806 | ML reconciliation | Medium | Baseline if available |
| U-Cast | https://openreview.net/forum?id=CCV9RqCCoQ | Latent hierarchical channel structure | High for latent hierarchy | Avoid latent hierarchy claims |
| Nixtla hierarchicalforecast/HINT | https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/hierarchical_forecasting.html | Practical implementations | Medium | Use for reproducible baselines |
| Robust statistics/classical measurement-error reconciliation | TBD deeper search | Reliability theory | Medium | Need stronger classical citation pass |

## Novelty Boundary

LedgerCast must focus on **noisy, revised, incomplete ledgers**. "Soft coherence" or "learned projection" alone is not novel. The key differentiator is reliability estimation and contamination-controlled evaluation.

## Uncertainty

Search found little top-venue neural work explicitly on contaminated hierarchy ledgers, but classical statistics may have robust reconciliation variants. This must be checked before M2.

## Novelty Confidence

**Medium.** Strong if M1 proves hard reconciliation hurts under realistic contamination; weak otherwise.
