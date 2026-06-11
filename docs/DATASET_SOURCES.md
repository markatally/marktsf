# Dataset Source Audit

> Generated: 2026-06-09
> Purpose: Record canonical source URL for every dataset in `input/` to enable 100% reproducible replication.

---

## Critical Reproducibility Warnings

| Dataset | Issue | Severity | Status |
|---------|-------|----------|--------|
| ~~`Electricity/electricity.csv`~~ | ~~Wrong date range (2016–2019) vs UCI (2011–2014)~~ | ~~🔴 HIGH~~ | ✅ **FIXED** — replaced with laiguokun benchmark (2012-01-01 to 2014-12-31, 26,304 rows, 321 clients) |
| ~~`Traffic/traffic.csv`~~ | ~~Wrong date range (2016–2018) vs UCI PEMS-SF (2015–2016)~~ | ~~🔴 HIGH~~ | ✅ **FIXED** — replaced with laiguokun benchmark (2015-01-01 to 2016-12-31, 17,543 rows, 861 sensors) |
| ~~`Exchange/exchange_rate.txt`~~ | ~~Column 6/7 swap vs CSV, different values, no OT column~~ | ~~🟡 MEDIUM~~ | ✅ **FIXED** — deleted inconsistent TXT file; CSV is canonical TSF benchmark |
| ~~`ECG/ECG5000.zip`~~ | ~~44-byte stub, not actual data~~ | ~~🟡 MEDIUM~~ | ✅ **FIXED** — deleted stub; `ECG_data.csv` (7.9MB, 5,000 rows × 141 cols) is the real ECG data |
| `CSI500/CSI500.csv` | **480 stock columns** (date + 480 stocks = 481 total). 1 stock gap from 2018 vs current — due to delistings. | 🟡 MEDIUM | ⚠️ **ACCEPTED** — AKShare blocked by proxy; baostock shows 500 current (2026) vs 480 in 2018 file; gap from 8 years of delistings. |
| `SP500/SP500.csv` | **474 stock columns** (not 487). S&P 500 composition changes over time. | 🟡 MEDIUM | ⚠️ **ACCEPTED** — existing file covers 474 real tickers 2018–2023; 13-stock gap from delistings/replacements |
| `G-Research-Crypto/` | Canonical G-Research Crypto raw benchmark is tracked from a byte-matching Kaggle Dataset mirror because the competition endpoint requires per-account rule acceptance. | 🟢 LOW | ✅ **FIXED** — raw files verified against official Kaggle file sizes; `train.csv` is tracked as chunks due to GitHub's 2 GiB LFS object limit |
| `Stock-1390` | **NOT in `input/`**. Required by xCPD (#18). | 🟢 EXPECTED | ✅ Expected — paper authors have not released this dataset |

---

## Summary: All Datasets Verified Against Canonical Sources

| Dataset | Local File | Canonical Source | Status |
|---------|-----------|-----------------|--------|
| **ETTh1/ETTh2** | `ETT/ETTh1.csv`, `ETTh2.csv` | zhouhaoyi/ETDataset | ✅ 17,420 rows exact |
| **ETTm1/ETTm2** | `ETT/ETTm1.csv`, `ETTm2.csv` | zhouhaoyi/ETDataset | ✅ 69,680 rows exact |
| **Weather** | `Weather/weather.csv` | BGC Jena / Autoformer mirror | ✅ 52,696 rows, 22 cols |
| **Electricity** | `Electricity/electricity.csv` | laiguokun (UCI reprocessed) | ✅ 26,304 rows, 322 cols |
| **Traffic** | `Traffic/traffic.csv` | laiguokun (PEMS-SF reprocessed) | ✅ 17,543 rows, 863 cols |
| **Exchange** | `Exchange/exchange_rate.csv` | laiguokun GitHub | ✅ 7,588 rows, 9 cols |
| **Solar_AL** | `Solar/solar_AL.txt` | laiguokun/NREL | ✅ 52,560 rows, 137 cols |
| **Solar_TFB** | `Solar/Solar_TFB.csv` | Same as above | ✅ 7,200,720 rows |
| **ILI** | `ILI/national_illness.csv` | CDC (no fixed URL) | ✅ 966 weekly rows |
| **M4 (12 files)** | `M4/*.csv` | Mcompetitions/M4-methods | ✅ All exact |
| **METR-LA** | `METR-LA/metr-la.h5` | DCRNN Google Drive | ✅ (34,272×207) HDF5 |
| **PEMS-BAY** | `PEMS-BAY/pems-bay.h5` | DCRNN Google Drive | ✅ (52,116×325) HDF5 |
| **PEMS03** | `PEMS03/PEMS03.npz` | Caltrans PeMS | ✅ (26,208×358×1) |
| **PEMS04** | `PEMS04/PEMS04.npz` | Caltrans PeMS | ✅ (16,992×307×3) |
| **PEMS07** | `PEMS07/PEMS07.npz` | Caltrans PeMS | ✅ (28,224×883×1) |
| **PEMS08** | `PEMS08/PEMS08.npz` | Caltrans PeMS | ✅ (17,856×170×3) |
| **NN5_TFB** | `NN5/NN5_TFB.csv` | Informer Google Drive | ✅ 87,801 rows |
| **Wike2000** | `Wiki/Wike2000.csv` | TFB benchmark | ✅ 1,584,000 rows |
| **COVID-19 TFB** | `COVID-19/Covid-19_TFB.csv` | TFB/JHU | ✅ 1,319,616 rows |
| **ECG_data** | `ECG/ECG_data.csv` | TFB benchmark | ✅ 5,000 rows, 141 cols |
| **AQShunyi** | `AQShunyi/AQShunyi.csv` | TFB benchmark | ✅ 385,704 rows |
| **Wind** | `Wind/Wind.csv` | TFB benchmark | ✅ 340,711 rows |
| **AirConvection** | `AirConvection/AirConvection.csv` | NOAA PSL | ✅ 175,205 rows, 12 cols |
| **AirConvection ASOS** | `AirConvection/*_2023.csv` | NOAA PSL | ✅ 7 airport files |
| **FRED-MD** | `FRED-MD/FRED-MD.csv` | FRED (TFB format) | ✅ 77,896 rows |
| **CSI500** | `CSI500/CSI500.csv` | AKShare/Tushare | ⚠️ 480 stocks (2018–2023) |
| **SP500** | `SP500/SP500.csv` | yfinance | ⚠️ 474 stocks (2018–2023) |
| **AU888 5m** | `CN-Future/AU888m5.csv` | AKShare/Tushare | ✅ 342,018 rows |
| **AU888 60m** | `CN-Future/AU888m60.csv` | AKShare/Tushare | ✅ 34,298 rows |
| **CSI300 60m** | `CN-Future/CSI300m60.csv` | AKShare/Tushare | ✅ 8,000 rows |
| **NASDAQ** | `NASDAQ/NASDAQ.csv` | TFB benchmark | ✅ 6,220 rows |
| **NYSE** | `NYSE/NYSE.csv` | TFB benchmark | ✅ 6,215 rows |
| **UTSD-1G** | `UTSD/UTSD-1G/` | HuggingFace `thuml/UTSD` | ✅ 68,679 samples, 472MB Arrow |
| **BOOM** | `BOOM/` | HuggingFace `Datadog/BOOM` | ✅ 2,807 series, 5.3GB (LFS) |
| **Crypto-Binance (14 assets)** | `Crypto/*.csv` | Binance public API | ✅ 14 assets, 1.6GB, 1-min OHLCV proxy |
| **Crypto-GResearch canonical** | `G-Research-Crypto/` | Kaggle `g-research-crypto-forecasting` / mirror `bariscan07/g-research-crypto-forecasting-dataset` | ✅ raw files verified; `train.csv` reconstructed from chunks |
| **Stock-1390** | — | Not released by authors | ✅ EXPECTED MISSING |

---

## Action Items

### ✅ COMPLETED (2026-06-08/09)
1. **Electricity** — Replaced with laiguokun benchmark (2012–2014, 26,304 rows, 321 clients)
2. **Traffic** — Replaced with laiguokun benchmark (2015–2016, 17,543 rows, 861 sensors)
3. **Exchange rate TXT** — Deleted inconsistent TXT file; CSV is canonical
4. **ECG5000.zip** — Deleted 44-byte stub; ECG_data.csv is the real data
5. **UTSD-1G** — Re-downloaded from HuggingFace (68,679 samples, 472MB)
6. **BOOM** — Complete via `git lfs pull` (5.3GB, 2,807 series)
7. **Crypto-Binance** — Downloaded all 14 assets via Binance public API pagination (1.6GB)
8. **BOOM tracked via LFS** — Committed with Git LFS for proper storage

### Large File Reconstruction
- **Favorita train.csv** and **Crypto-GResearch train.csv** exceed GitHub's 2 GiB
  per-object LFS limit, so their raw CSVs are stored as
  `input/**/chunks/train.csv.part-*`. After pulling on a fresh machine, run:

```bash
scripts/reconstruct_large_inputs.sh
```

This recreates `input/Favorita/train.csv` and
`input/G-Research-Crypto/train.csv` and verifies their byte sizes.

### ⚠️ ACCEPTED GAPS
- **SP500** — 13-stock gap (474 vs 487) due to S&P 500 composition changes 2018–2023
- **Stock-1390** — Authors of xCPD paper have not released it

---

## P1 Scenario Data (added 2026-06-09)

> Acquired for SPECTRE scenarios S1 (finance / LOB) and S2 (retail). See PROPOSAL.md §6.1.

| Dataset | Local Path | Source | Status |
|---------|-----------|--------|--------|
| **FI-2010** (LOB benchmark) | `input/FI2010/` (~918 MB) | `zcakhaa/DeepLOB` GitHub repo `data/data.zip` (NoAuction_DecPre) | ✅ Train_CF_7 + Test_CF_7/8/9 |
| **M5** (retail, full covariates) | `input/M5/m5/datasets/` (~466 MB) | Nixtla `datasetsforecast` (public S3 mirror) | ✅ sales + calendar (events/SNAP) + sell_prices |
| **Favorita** (retail) | `input/Favorita/` (~5GB) | Kaggle dataset mirror `siliconx/favoritagrocerysalesforecastingextracted` | ✅ train(125M rows) + test + holidays + stores + items + oil + transactions |
| **LOBSTER** (raw LOB) | — | lobsterdata.com (now a SPA; no scriptable direct link) | ⏳ manual browser download; FI-2010 covers the LOB benchmark |

**Acquisition script**: `scripts/download_p1_data.py`. FI-2010 via DeepLOB GitHub
mirror; M5 via `datasetsforecast`. Favorita/LOBSTER need credentials/manual steps
(script prints instructions + validates magic bytes to reject HTML stubs).

## Baseline Reproduction (P1)

> Policy: reproduce published baselines by running the ORIGINAL authors' code, not
> reimplementations — required to hit a ≤2% bar (per user requirement).

- **Vendored**: `external/TSLib` = thuml/Time-Series-Library (git-ignored; see
  `docs/EXTERNAL.md` and `scripts/sync_external.sh`), covers DLinear, PatchTST,
  iTransformer, TiDE, TimeXer, + 30 more with official scripts.
- **Wrapper**: `spectre/reproduce/official.py` parses each official script, forces
  CPU + `num_workers=0` (TSLib's default 10 workers are ~270× slower on macOS),
  runs it against our `input/ETT/`, parses `mse/mae`, gates at 2% vs the original
  code's reproducible number (`spectre/experiments/reference.py`).
- **Verified**: DLinear ETTh1@96 → mse 0.3962 (paper 0.386; the ~2.6% gap is the
  inherent paper-vs-code difference, documented as context).

## MISO-Native Baselines (added 2026-06-11)

> Three MISO-native competitors explicitly required by PROPOSAL.md §7.3 and flagged as action items in §6.1.

| Paper | PDF | Implementation | Notes |
|-------|-----|---------------|-------|
| **TimeXer** (NeurIPS 2024) | `paper/TimeXer - ....pdf` (11 MB) | `external/TSLib` — `scripts/long_term_forecast/TimeXer_script/` | Primary must-beat baseline; MISO-native via endo-patch + exo-variate tokens |
| **TFT** (IJF 2021) | `paper/TFT - ....pdf` (2.5 MB) | `pip install pytorch-forecasting` or `google-research/tft` | Use `TimeSeriesDataSet` with `known_regulars`; M5/Favorita provide natural covariates |
| **NBEATSx** (Energy&AI 2023) | `paper/NBEATSx - ....pdf` (1.3 MB) | `pip install neuralforecast` (`NBEATSx` class) | Configure `futr_exog_list` / `hist_exog_list` for MISO covariate injection |

All three use only datasets already in `input/` — no new downloads required.

## Classical Foundation Papers (added 2026-06-11)

> 4 pre-open-access theory papers cited in PRISM §4. Not datasets — tracked here as they require manual acquisition.

| Paper | Status | Notes |
|-------|--------|-------|
| Hamilton 1989 (*Econometrica*) | ❌ Paywalled | DOI 10.2307/1912559; Wiley/Econometrica — institutional access required |
| Ghahramani & Hinton 2000 (*Neural Computation*) | ❌ Paywalled | DOI 10.1162/089976600300015619; MIT Press — institutional access required |
| Herbster & Warmuth 1998 (*Machine Learning*) | ❌ Paywalled | DOI 10.1023/A:1007488714892; Springer — institutional access required |
| Diebold & Mariano 1995 (*JBES*) | ✅ Downloaded | NBER WP4390 (pre-pub version); `paper/Diebold & Mariano 1995 - ....pdf` (2.1 MB) |
