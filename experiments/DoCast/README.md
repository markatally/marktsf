# DoCast — Do-Operator Forecasting

Intervention-valid scenario forecasting for MISO time series with controllable covariates
(prices, promotions). Single-model, train-once, architecture-agnostic: a structural
response head + sequentially orthogonalized (DML-style) training objective on top of any
existing backbone, plus the first Scenario Validity Audit of SOTA MISO forecasters.

- **Proposal (single source of truth)**: [docs/PROPOSAL.md](docs/PROPOSAL.md) — v1.0, pre-G0.
- **Status**: G0 (prior-art sweep) not yet run; no code yet. First steps in PROPOSAL §9.
- **Battlefields**: M5 (`input/M5/`), Favorita (`input/Favorita/`), released semi-synthetic suite.
- **Disjointness**: by construction non-overlapping with `experiments/PRISM/` — no regime
  tracking, no expert routing / MoE, no CI/CD gating, no drift or test-time adaptation
  (PROPOSAL §6.4).
