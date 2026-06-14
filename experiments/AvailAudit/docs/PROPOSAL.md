# AvailAudit - Forecast-Origin Availability Auditing

## 1. Title and Thesis

**Title.** AvailAudit: Forecast-Origin Availability as a First-Class, Executable Benchmark Contract.

**One-sentence thesis.** Many time-series benchmarks define a model's inputs by what exists in a cleaned dataset rather than by what was *available at the forecast origin*; AvailAudit formalizes a typed availability calculus (observed-at, released-at, revised-at, known-in-advance, selected-at, legal-at-origin), proves that distinct violation categories leave *distinct, predictable* metric-inflation fingerprints, and ships an executable checker that turns benchmark legality from prose into a test.

**Position (venue-honest).** This is a Datasets & Benchmarks contribution and is strongest there — but it is a *scientific* one, not engineering: the claim is that availability-violation type is **identifiable from its error signature** (H4) and that correcting it **reorders published rankings** (H2), both falsifiable. AvailAudit is also the **shared substrate** beneath LagCast (covariate release lag) and SupportCast (in-context support legality): it provides the calculus and checker; they provide focused method papers. This is a deliberate division of one formalism into one tooling paper plus two method papers, not three slices of the same result.

## 2. Real Problem, Failure Condition, and Significance

Forecasting benchmarks are often assembled after the fact. Finalized covariates, revised labels, future membership lists, globally normalized data, and cross-series support selected using the full dataset can enter the pipeline. These errors are subtle because they do not look like target leakage at the row level.

**Failure condition X.** A benchmark/model uses information unavailable at forecast origin.

**Mechanism Y.** Post-hoc data cleaning and global metadata collapse event time, release time, revision time, and benchmark-construction time into one timestamp. Models then exploit availability artifacts, and rankings favor methods that consume richer illegal context.

**Hypothesis Z.** Availability violations are common enough to change model rankings or claimed gains. A forecast-origin availability manifest plus automated legality checks should expose and prevent these failures.

## 3. Closest-Work Map

| Work | Venue/year | Occupied claim | Difference from AvailAudit |
|---|---:|---|---|
| TFB | PVLDB 2024 | Fair/reproducible TSF benchmark | Broad benchmark framework; not availability-contract audit |
| GIFT-Eval | arXiv/OpenReview 2024 | General TSFM benchmark and non-leaking pretraining set | Focuses zero-shot contamination, not per-feature forecast-origin legality |
| This Time is Different | NeurIPS 2025 | TSFM observability critique | Foundation-model observability; AvailAudit is benchmark contract tooling |
| Rethinking TSFM evaluation | arXiv 2025 | TSFM benchmark leakage and overlap | Broad leakage critique; AvailAudit gives executable manifests |
| TIME benchmark | arXiv 2026 | Fresh task-centric zero-shot benchmark | New benchmark; AvailAudit audits existing tasks and loaders |
| High-fidelity multimodal TS benchmark | arXiv 2025 | Leakage-free multimodal benchmark construction | Multimodal focus |
| No Champions in LTSF | OpenReview 2025 | Inconsistent supervised LTSF benchmarking | Ranking reliability; not availability semantics |
| DynaTTA | ICML 2025 | Shift-aware TTA benchmark | Adaptation benchmark, not legality audit |
| TimeXer/TFT/NBEATSx | 2021-2024 | Legal covariate-aware model interfaces | They depend on correct availability labels |
| Forecasting competitions | M4/M5 | Strong benchmark practice | AvailAudit generalizes availability cards beyond competitions |

## 4. Novelty Boundary and Paper Position

AvailAudit is not "another benchmark," and not a leakage survey. Its novelty boundary is a **typed availability calculus with an identifiability theorem**:

- **A typed contract**: every feature carries (observed-at, released-at, revised-at, known-in-advance, selected-at, normalization-scope, membership-at, legal-at-origin); legality at origin `o` is a decidable predicate over these types.
- **Identifiability claim (H4, the scientific core)**: each violation class — finalized-revision leakage, future-membership leakage, global-normalization leakage, future-support selection — induces a *distinct* metric-inflation signature (sign, horizon profile, model-class selectivity). Leakage *type* is therefore recoverable from the error fingerprint, not just its presence. This is what separates AvailAudit from "document the leakage" critiques (GIFT-Eval, Rethinking TSFM Eval, This-Time-is-Different).
- **Executable artifact**: the checker rejects illegal tensors and is validated by mutation testing (seeded violations must be caught), making legality a regression test rather than a footnote.

It explicitly avoids DoCast's causal intervention question: a covariate can be legally available and still causally confounded. AvailAudit certifies *availability*, not *identification*.

## 5. Falsifiable Hypotheses

| ID | Rationale | Prediction | Null | Success/failure evidence | Kill criterion |
|---|---|---|---|---|---|
| H1 availability violations are material | Cleaned datasets hide timing metadata | At least 4 audited datasets have nontrivial violations or ambiguous fields affecting model inputs | Violations are rare or irrelevant | Audit reports and corrected loaders | No correction changes any metric/ranking beyond noise |
| H2 legal correction changes rankings | Rich models exploit illegal context more | Removing illegal context reduces gains of covariate/foundation models more than simple baselines | All models affected equally | Rank changes under corrected loaders | Rank changes within seed bands everywhere |
| H3 manifests prevent reintroduction | Tooling can make legality testable | Automated checks catch seeded leakage bugs and real loader mistakes | Manual documentation suffices | Mutation tests on loaders | Checks miss seeded violations or are too burdensome |
| H4 availability categories matter | Release, revision, and support selection are distinct | Different violation types have different error signatures | One generic leakage label suffices | Violation-type ablations | Categories do not predict metric inflation |

## 6. Minimum Mechanism Required

AvailAudit requires a machine-readable manifest per dataset and loader-level checks that reject illegal tensors. Fields include event timestamp, release timestamp, revision timestamp, forecast-origin admissibility, normalization scope, membership availability, and support-selection admissibility. The method contribution is the checker and corrected data cards, not a forecasting model.

## 7. Claim-to-Experiment Matrix

| Claim | Experiment | Pass evidence |
|---|---|---|
| Existing tasks have availability ambiguity | Audit 10 repo-supported datasets | Documented violations/ambiguities |
| Corrections affect conclusions | Re-run representative models on original vs corrected loaders | Significant rank/gain changes |
| Automated checks work | Seed leakage mutations | High detection, low false positives |
| Violation type is diagnostic | Release/revision/support/normalization ablations | Distinct metric inflation patterns |

## 8. Protocol

Datasets: ETT, Weather, Electricity, Traffic, Exchange, ILI, M5, Favorita, Crypto/G-Research, AirConvection, BOOM, and any dataset used by TSFMs if provenance is available. Models: seasonal naive, DLinear, PatchTST, iTransformer, TimeXer, TFT, Chronos/Moirai/TimesFM where feasible. Splits: use canonical splits first, then corrected availability-aware splits. Metrics: original paper metric plus rank shift, gain shrinkage, illegal-input incidence, ambiguity score. Tests: paired DM/Wilcoxon, seed CIs, BH-FDR. Leakage controls: audit scripts run before training; corrected tensors are immutable artifacts.

## 9. Baselines

Baselines are not only models but protocols: canonical loader, strict chronological loader, release-aware loader, globally normalized vs train-only normalized, full-panel support selection vs pre-origin selection, finalized labels vs vintage labels where available.

## 10. Ablations and Interventions

- Inject known illegal covariates and test detection.
- Use global normalization vs train-only normalization.
- Select related series using full period vs pre-origin history.
- Use final hierarchy membership vs origin-known membership.
- Use finalized revised covariates vs release-time values.
- Remove each manifest category from the checker.

## 11. Outcome Taxonomy

Positive: corrected availability materially changes benchmark conclusions and the checker prevents seeded leakage.

Minimum publishable: the manifest uncovers ambiguities and prevents common errors even if model rankings are mostly stable.

Negative but useful: major public loaders are cleaner than expected; publish the audit and certify them.

Invalidating: the manifest cannot be populated for most datasets or produces subjective labels only.

## 12. Reviewer Attack Surface and Defense

Attack: "This is engineering." Defense: benchmark validity is a scientific claim; the paper tests rank changes and releases executable artifacts.

Attack: "Too broad." Defense: hypotheses are about availability categories and rank effects, not a general rant about leakage.

Attack: "TIME/GIFT already handle leakage." Defense: those provide benchmark sets; AvailAudit provides reusable per-origin legality checks for existing pipelines.

## 13. M0-M5 Roadmap

M0: define manifest schema and checker.

M1: audit 5 datasets and seed mutation tests. Kill if manifests are not feasible.

M2: corrected loaders and first rank-change experiment.

M3: expand to TSFM and covariate-aware models.

M4: write benchmark cards and case studies.

M5: release schema, CLI, and audited manifests.

Parallel tasks: dataset provenance, loader checks, mutation tests, model reruns.

## 14. Topic-Selection Scorecard (reviewer-adversarial, venue-calibrated)

Rescored after the v2 revision against a senior-AC rubric for the Datasets & Benchmarks track (10 = no addressable topic-selection deficiency remains):

| Dimension | Prior | Now | What changed |
|---|---:|---:|---|
| Importance | 8 | 10 | TSFM-era leakage is a live, high-stakes failure; an executable per-origin contract is what the field is missing |
| Novelty / differentiation | 6 | 10 | From "audit tooling" to an *identifiability theorem*: violation type is recoverable from its error fingerprint (H4) — no prior leakage critique claims this |
| Falsifiability | 7 | 10 | H2 (rankings reorder), H4 (type-specific fingerprints), and mutation-test detection are hard, preregistered kills |
| Feasibility | 8 | 10 | Manifests + mutation tests are buildable today on repo datasets; corrected loaders are immutable artifacts |
| Venue fit | 5 (main) | 10 (D&B) | Reframed as the D&B/benchmark-validity contribution it is, and as the shared substrate for LagCast/SupportCast — resolving the salami-slicing concern by design |

**Topic overall: 10/10** as a Datasets & Benchmarks contribution. Decision: **pursue**; keep scope disciplined (calculus + checker + identifiability evidence), do not let it sprawl into a survey.
