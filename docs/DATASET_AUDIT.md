# Dataset Audit — 18 TSF Papers

> Generated 2026-06-07; updated 2026-06-11 (added TimeXer, TFT, NBEATSx). Verified against each paper's PDF and official code repository.

## Status Summary

| Status | Count | Datasets |
|--------|-------|---------|
| ✅ Downloaded & verified | 45 items | ETT×4, Weather, Electricity, Traffic, Exchange×2, ILI, Solar×2, PEMS03-08×4 (npz), METR-LA×2, PEMS-BAY×2, M4×12, NN5×3, Wike2000, COVID-19×2, ECG, AQShunyi, AQWan, Wind, SP500, CSI500 (480 stocks), AirConvection (5 stations, 10 vars), UTSD-1G (84 files), BOOM (2.7GB, 2807 dirs), Crypto-Binance (14 assets, 1.6GB) |
| ✅ Downloaded via verified mirror | 1 item | Crypto-GResearch canonical (`g-research-crypto-forecasting`) — raw files verified against official Kaggle file sizes; `train.csv` stored as chunks because GitHub LFS rejects >2 GiB objects |
| 🚫 Not yet released | 1 item | Stock-1390 (xCPD) |

---

## Per-Paper Dataset Requirements

### 1. SEMPO (NeurIPS 2025) — `mala-lab/SEMPO`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard split: 12/4/4 months |
| Weather, Electricity, Traffic | ✅ | Standard split |
| UTSD (pre-training) | ✅ | `UTSD/` — UTSD-1G Arrow format (84 files, 0.49GB); for full numpy, see manual |

### 2. This Time is Different (NeurIPS 2025) — `DataDog/toto`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard |
| Weather, Electricity | ✅ | Standard |
| BOOM benchmark | ✅ | `BOOM/` — 2.7GB, 2,807 series dirs, 16,860 Arrow files; HuggingFace `Datadog/BOOM` |

### 3. Time Tracker (ArXiv 2025) — code not public
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard |
| Weather, Electricity, Traffic | ✅ | Standard |
| UTSD (pre-training) | ✅ | `UTSD/` — UTSD-1G Arrow format (84 files, 0.49GB) |

### 4. How Biased is TSF (ArXiv 2025) — `IbramMedhat/How-Biased-is-...`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Split: 7:1:2 |
| Weather, Electricity, Traffic | ✅ | Split: 7:1:2 |
| Solar | ✅ | `solar_AL.txt` |

### 5. Partial Channel Dependence (ICASSP 2026) — `YonseiML/pcd`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard |
| Weather, Electricity, Traffic, Exchange, ILI | ✅ | Standard |
| PEMS03/04/07/08 | ✅ | `.npz` format, iTransformer pack |
| Solar | ✅ | `solar_AL.txt`, NREL Alabama |

### 6. Channel Strategy Survey (ArXiv 2025) — `decisionintelligence/CS4TS`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard |
| Traffic, Exchange | ✅ | Standard |
| PEMS04, PEMS08, PEMS-BAY | ✅ | TFB CSV format (`*_TFB.csv`) |
| AQWan (AQ Wan station) | ✅ | `AQShunyi/AQWan.csv`, TFB pack |
| Solar (TFB format) | ✅ | `Solar/Solar_TFB.csv`, TFB pack |

### 7. Routing Channel-Patch / xCPD (ArXiv 2026) — `Clearloveyuan/xCPD`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard (long-term) |
| Weather, Electricity, Traffic, Exchange, ILI | ✅ | Standard |
| Solar | ✅ | `solar_AL.txt` |
| M4 | ✅ | Short-term forecasting (all 6 splits) |
| Stock-1390 (Chen et al., 2024) | 🚫 | Not released; see `Clearloveyuan/xCPD` |

### 8. TimeBridge (ICML 2025) — `Hank0626/TimeBridge`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard |
| Weather, Electricity, Traffic | ✅ | Standard |
| Solar | ✅ | `solar_AL.txt` |
| PEMS03/04 | ✅ | `.npz` format (short-term) |
| CSI500 (502 Chinese stocks 2018-2023) | ✅ | `CSI500/CSI500.csv` — 480 stocks, 1457 rows (2018–2023), via AKShare/Sina |
| S&P500 (487 US stocks 2018-2023) | ✅ | `SP500/SP500.csv` (475 stocks via yfinance) |

> **Note on S&P500**: Downloaded 475 stocks (vs paper's 487). Minor discrepancy due to ticker changes/delistings since 2023. Use `yfinance` to re-download with updated constituent list.

> **Note on TimeBridge financial code**: The financial experiments (CSI500/S&P500) are NOT implemented in the public repo `Hank0626/TimeBridge`. The repo only covers long-term forecasting with ETT/Weather/Electricity/Traffic/Solar/PEMS. You may need to request the financial experiment code from the authors.

### 9. DTAF (ArXiv 2025 / AAAI 2026) — `decisionintelligence/DTAF`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | TFB framework |
| Weather, Electricity, Traffic, ILI | ✅ | TFB framework |
| Wike2000 | ✅ | `Wiki/Wike2000.csv`, TFB pack |
| Covid-19 (TFB 948-channel) | ✅ | `COVID-19/Covid-19_TFB.csv`, TFB pack |
| NN5 | ✅ | `NN5/NN5_TFB.csv`, TFB pack |
| AQShunyi | ✅ | `AQShunyi/AQShunyi.csv`, TFB pack |

> **DTAF uses TFB evaluation framework** (`decisionintelligence/TFB`), not the standard TSLib. Use `ts_benchmark` runner: `python scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list "ETTh1.csv"`.

### 10. FreDF (ICLR 2025) — `Master-PLC/FreDF`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard |
| Weather, Electricity, Traffic | ✅ | Standard |
| PEMS03, PEMS08 | ✅ | `.npz` format |
| M4 | ✅ | Short-term forecasting (all 6 splits) |

### 11. FreTS (NeurIPS 2023) — `aikunyi/FreTS`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1, ETTm1 | ✅ | Split: 7:2:1 (differs from standard 7:1:2!) |
| Weather, Electricity, Traffic, Exchange | ✅ | Split: 7:2:1 |
| Solar | ✅ | `solar_AL.txt`, NREL Alabama |
| covid.csv (CA hospitalization) | ✅ | `COVID-19/covid_california.csv` |
| ECG_data.csv (140-lead ECG) | ✅ | `ECG/ECG_data.csv` |
| METR-LA | ✅ | `METR-LA/metr-la.h5` |

> **FreTS uses 7:2:1 split** (not 7:1:2). This is important for replication — results will differ if using the wrong split.

### 12. Time-SSM (ICML 2025) — code not public
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard |
| Weather, Electricity, Traffic, Exchange | ✅ | Standard |
| Crypto-Binance (14-asset, 1-min OHLCV proxy) | ✅ | `Crypto/*.csv` — 14 assets, 1.6GB, via Binance public API (2017–2022, 1-min OHLCV). This is **not** the canonical G-Research schema. |
| Crypto-GResearch canonical (14 assets, 8-ish trading columns + Target) | ✅ | `G-Research-Crypto/` — raw files from Kaggle competition schema, downloaded via byte-matching mirror `bariscan07/g-research-crypto-forecasting-dataset`; `VERIFY.json` confirms official file sizes, columns, 14 assets, and row counts. `train.csv` is reconstructed from `chunks/train.csv.part-*`. |
| Air Convection (NOAA 2023, 10-var, 15-min) | ✅ | `AirConvection/` — NOAA CRN 5-min → 15-min agg, 5 stations, 35,041 rows/station, 10 vars |

> **Time-SSM code not publicly released** (arXiv 2405.16312). The paper references `alxndrTL/mamba.py` only for the SSM building block. Contact authors for full code.

### 13. Proceed (KDD 2025) — `SJTU-DMTai/OnlineTSF`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh2, ETTm1 | ✅ | Standard |
| Weather, Electricity, Traffic | ✅ | Standard |

### 14. DynaTTA (ICML 2025) — `shivam-grover/DynaTTA`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1, ETTh2, ETTm1 | ✅ | Split: **60:20:20** (differs from standard!) |
| Weather, Electricity, Exchange, ILI | ✅ | Split: 60:20:20 |

> **DynaTTA uses 60:20:20 split**. This is different from both the standard 7:1:2 and FreTS's 7:2:1. Use DynaTTA's own data loader.

### 15. Tackling Generalization via Concept Drift (ArXiv 2026) — `AdityaLab/ShifTS`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard |
| Weather, Electricity, Traffic, Exchange, ILI | ✅ | Standard |

### 16. TimeXer (NeurIPS 2024) — `thuml/Time-Series-Library`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard split |
| Weather, Electricity, Traffic, Exchange, ILI | ✅ | Standard split |
| Solar | ✅ | `solar_AL.txt` |

> **TimeXer is included in TSLib** (`external/TSLib`): run via `scripts/long_term_forecast/TimeXer_script/`. For PRISM MISO experiments (S1/S2), TimeXer is used as the primary must-beat baseline; it supports exogenous variables natively via its `enc_in` / `dec_in` MISO interface.

### 17. TFT (IJF 2021) — `google-research/tft` / `pytorch-forecasting`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard split; use as MISO-ized benchmark |
| Weather, Electricity, Traffic | ✅ | Standard split |
| M5 | ✅ | `input/M5/` — natural known-future covariates (calendar / SNAP / prices) |
| Favorita | ✅ | `input/Favorita/` — natural known-future covariates (promotions, holidays, oil) |

> **TFT implementation**: use `pytorch-forecasting` (`pip install pytorch-forecasting`) or the official `google-research/tft` repo. TFT expects `known_future` covariates in its `TimeSeriesDataSet` format — M5 and Favorita provide these naturally. For symmetric benchmarks, use a single-target MISO setup (ETT→OT column as target). No new datasets required.

### 18. NBEATSx (Energy&AI 2023) — `Nixtla/neuralforecast`
| Dataset | Status | Notes |
|---------|--------|-------|
| ETTh1/h2, ETTm1/m2 | ✅ | Standard split |
| Weather, Electricity, Traffic | ✅ | Standard split |
| M5 | ✅ | `input/M5/` — exogenous covariates supported via `futr_exog_list` |
| Favorita | ✅ | `input/Favorita/` — promotions / holiday flags as exogenous |

> **NBEATSx implementation**: use `neuralforecast` (`pip install neuralforecast`), which provides `NBEATSx` natively. Configure `futr_exog_list` / `hist_exog_list` to pass MISO covariates. No new datasets required.

---

## Critical Split Differences

Different papers use different train/val/test splits — **using the wrong split will change results**:

| Split | Papers | Train:Val:Test |
|-------|--------|----------------|
| **Standard (Autoformer)** | Most papers | ~70:10:20 (fixed boundaries) |
| **FreTS 7:2:1** | FreTS | 70:20:10 |
| **DynaTTA 60:20:20** | DynaTTA | 60:20:20 |
| **TFB rolling** | DTAF, Channel Survey | Rolling window forecast |

For ETT specifically, the Autoformer standard uses absolute timestep boundaries:
- ETTh: train [0:8640], val [8640:11520], test [11520:17421]
- ETTm: train [0:34560], val [34560:46080], test [46080:69680]

---

## Data Source Provenance

| Dataset | Canonical Source | Format | Papers |
|---------|-----------------|--------|--------|
| ETT×4 | thuml/Autoformer GDrive | CSV (date+7cols) | All |
| Weather | thuml/Autoformer GDrive | CSV (date+21cols) | All |
| Electricity | UCI / thuml | CSV (date+321cols) | All |
| Traffic | PEMS-BAY processed / thuml | CSV (date+862cols) | All |
| Exchange | LSTNet / thuml | CSV (date+8cols) | Most |
| ILI | CDC / thuml | CSV (date+7cols) | Several |
| Solar | NREL Alabama | TXT (52560×137) | Several |
| PEMS03/04/07/08 | iTransformer GDrive | NPZ (T×N×C) | Several |
| METR-LA | DCRNN paper / thuml | HDF5 (34272×207) | FreTS |
| PEMS-BAY | DCRNN paper / thuml | HDF5 (52116×325) | FreTS |
| M4 | M-competitions GitHub | CSV | FreDF, Routing |
| NN5 | Monash Zenodo | TSF + TFB CSV | FreDF, DTAF |
| Wike2000 | TFB GDrive | CSV (1427×2001) | DTAF |
| Covid-19 (TFB) | TFB GDrive | CSV (1392×949) | DTAF |
| covid_california | FreTS GitHub | CSV (CA hospitalization) | FreTS |
| ECG_data | FreTS GitHub | CSV (140-lead) | FreTS |
| AQShunyi/AQWan | TFB GDrive | CSV (Beijing AQ) | DTAF, Channel Survey |
| Wind | TFB GDrive | CSV | Channel Survey |
| SP500 | Yahoo Finance (yfinance) | CSV (475 stocks) | TimeBridge |
| CSI500 | investing.com / AKShare | CSV (502 stocks) | TimeBridge |
| Crypto-Binance | Binance public API | CSV (14 assets, 1min, 1.6GB) | PRISM proxy / high-frequency extension |
| Crypto-GResearch canonical | Kaggle `g-research-crypto-forecasting` | CSV (`timestamp`, `Asset_ID`, `Count`, `Open`, `High`, `Low`, `Close`, `Volume`, `VWAP`, `Target`) | Time-SSM / KRNO-aligned canonical benchmark |
| Air Convection | NOAA PSL (scraped) | CSV (10 vars, 15min) | Time-SSM |
| UTSD | HuggingFace thuml/UTSD | NPY (7 domains) | SEMPO, Time Tracker |
| BOOM | HuggingFace Datadog/BOOM | Arrow (350M obs) | This Time is Different |
| Stock-1390 | Chen et al., 2024 (TBD) | CSV (1390 stocks) | Routing/xCPD |

---

## Manual Download Instructions

### CSI500 (for TimeBridge)
```python
pip install akshare
import akshare as ak
import pandas as pd

# Get CSI500 component list
index = ak.index_stock_cons(symbol="000905")  # CSI500 = 000905; CSI300 (Hushen 300) = 000300
tickers = index['品种代码'].tolist()  # 品种代码 = ticker code

# Download 2018-2023 daily data for each stock
dfs = {}
for ticker in tickers:
    try:
        df = ak.stock_zh_a_hist(symbol=ticker, period="daily", 
                                 start_date="20180101", end_date="20231231",
                                 adjust="hfq")  # 后复权 (backward-adjusted)
        dfs[ticker] = df.set_index('日期')['收盘']  # 日期 = date, 收盘 = close price (AKShare column names)
    except:
        pass

# Combine and save
close = pd.DataFrame(dfs).reset_index()
close.rename(columns={'index': 'date'}, inplace=True)
close.to_csv("input/CSI500/CSI500.csv", index=False)
```

### Crypto (for Time-SSM / KRNO)
Two variants are tracked:

1. ✅ **Crypto-Binance proxy** — `input/Crypto/*.csv` (14 assets, 1.6GB, 2017–2022, 1-min OHLCV via Binance public API). Use `scripts/download_crypto.py` to re-download if needed.
2. ✅ **Crypto-GResearch canonical** — `input/G-Research-Crypto/` from Kaggle `g-research-crypto-forecasting`, expected columns `timestamp, Asset_ID, Count, Open, High, Low, Close, Volume, VWAP, Target`. The raw files are downloaded via the byte-matching Kaggle Dataset mirror `bariscan07/g-research-crypto-forecasting-dataset` and verified by:

```bash
python scripts/download_gresearch_crypto.py --skip-download
scripts/reconstruct_large_inputs.sh
```

### Air Convection (for Time-SSM)
- Time-SSM code not publicly released; contact authors at arXiv:2405.16312
- NOAA PSL data: https://www.psl.noaa.gov/data/gridded/
- Variables: air humidity, pressure, convection characteristics (10 vars, 15-min, 2023)

### UTSD (for SEMPO/Time Tracker pre-training)
```python
# Option A: HuggingFace datasets library (Arrow format)
from datasets import load_dataset
d = load_dataset("thuml/UTSD", "UTSD-1G")  # or UTSD-2G, UTSD-4G, UTSD-12G

# Option B: NumPy format from Tsinghua Cloud (recommended for SEMPO)
# Manual download from: https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
# Save to: input/UTSD/utsd.npy
```

### BOOM (for This Time is Different)
```python
from datasets import load_dataset
d = load_dataset("Datadog/BOOM")
d.save_to_disk("input/BOOM")
```
