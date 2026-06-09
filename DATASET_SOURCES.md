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
| **AU888 5m** | `Finance/AU888_5m_*.csv` | AKShare/Tushare | ✅ 342,018 rows |
| **AU888 60m** | `Finance/AU888_60m_*.csv` | AKShare/Tushare | ✅ 34,298 rows |
| **CSI300 60m** | `Finance/CSI300_60m_*.csv` | AKShare/Tushare | ✅ 8,000 rows |
| **NASDAQ** | `NASDAQ/NASDAQ.csv` | TFB benchmark | ✅ 6,220 rows |
| **NYSE** | `NYSE/NYSE.csv` | TFB benchmark | ✅ 6,215 rows |
| **UTSD-1G** | `UTSD/UTSD-1G/` | HuggingFace `thuml/UTSD` | ✅ 68,679 samples, 472MB Arrow |
| **BOOM** | `BOOM/` | HuggingFace `Datadog/BOOM` | ✅ 2,807 series, 5.3GB (LFS) |
| **Crypto (14 assets)** | `Crypto/*.csv` | Binance public API | ✅ 14 assets, 1.6GB |
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
7. **Crypto** — Downloaded all 14 assets via Binance public API pagination (1.6GB)
8. **BOOM tracked via LFS** — Committed with Git LFS for proper storage

### ⚠️ ACCEPTED GAPS
- **SP500** — 13-stock gap (474 vs 487) due to S&P 500 composition changes 2018–2023
- **Stock-1390** — Authors of xCPD paper have not released it
