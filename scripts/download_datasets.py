#!/usr/bin/env python3
"""
Download all benchmark datasets referenced in the 15 TSF papers.
Run from the repo root: python scripts/download_datasets.py
"""

import os
import sys
import zipfile
import urllib.request
import shutil
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")


def download(url, dest, desc=""):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  SKIP (exists): {os.path.basename(dest)}")
        return True
    print(f"  Downloading {desc or os.path.basename(dest)} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"  OK: {os.path.basename(dest)} ({os.path.getsize(dest):,} bytes)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_gdown(gdrive_id, dest, desc=""):
    """Download from Google Drive using gdown."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  SKIP (exists): {os.path.basename(dest)}")
        return True
    try:
        import gdown
    except ImportError:
        print("  FAIL: gdown not installed. Run: pip install gdown")
        return False
    print(f"  Downloading {desc} from Google Drive ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        gdown.download(id=gdrive_id, output=dest, quiet=False)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  OK: {os.path.basename(dest)} ({os.path.getsize(dest):,} bytes)")
            return True
    except Exception as e:
        print(f"  FAIL: {e}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. STANDARD DATASETS (thuml/Autoformer + iTransformer Google Drive pack)
#    Papers: ALL 15 papers use ETT, Weather, Electricity, Traffic; most use Exchange, ILI
# ─────────────────────────────────────────────────────────────────────────────
def download_standard():
    print("\n=== Standard Datasets (thuml/Autoformer + iTransformer packs) ===")
    # Source: thuml Autoformer pack (Google Drive)
    TMP_A = "/tmp/thuml_autoformer.zip"
    TMP_I = "/tmp/iTransformer_data.zip"

    # thuml/Autoformer pack: ETT, Weather, Electricity, Traffic, Exchange, ILI
    if not (os.path.exists(TMP_A) and os.path.getsize(TMP_A) > 1_000_000):
        download_gdown("1l4c00GKMVHmNV4Jg5HdX8FmWD3iyKUK9", TMP_A, "thuml Autoformer dataset pack")

    # iTransformer pack: ETT, Weather, Electricity, Traffic, Solar, PEMS03/04/07/08
    if not (os.path.exists(TMP_I) and os.path.getsize(TMP_I) > 10_000_000):
        download_gdown("1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot", TMP_I, "iTransformer dataset pack")

    extract_map = {
        TMP_A: {
            "ETT-small/ETTh1.csv":      f"{BASE}/ETT/ETTh1.csv",
            "ETT-small/ETTh2.csv":      f"{BASE}/ETT/ETTh2.csv",
            "ETT-small/ETTm1.csv":      f"{BASE}/ETT/ETTm1.csv",
            "ETT-small/ETTm2.csv":      f"{BASE}/ETT/ETTm2.csv",
            "weather/weather.csv":      f"{BASE}/Weather/weather.csv",
            "electricity/electricity.csv": f"{BASE}/Electricity/electricity.csv",
            "traffic/traffic.csv":      f"{BASE}/Traffic/traffic.csv",
            "exchange_rate/exchange_rate.csv": f"{BASE}/Exchange/exchange_rate.csv",
            "illness/national_illness.csv": f"{BASE}/ILI/national_illness.csv",
        },
        TMP_I: {
            "iTransformer_datasets/PEMS/PEMS03.npz": f"{BASE}/PEMS03/PEMS03.npz",
            "iTransformer_datasets/PEMS/PEMS04.npz": f"{BASE}/PEMS04/PEMS04.npz",
            "iTransformer_datasets/PEMS/PEMS07.npz": f"{BASE}/PEMS07/PEMS07.npz",
            "iTransformer_datasets/PEMS/PEMS08.npz": f"{BASE}/PEMS08/PEMS08.npz",
            "iTransformer_datasets/Solar/solar_AL.txt": f"{BASE}/Solar/solar_AL.txt",
        }
    }

    for zippath, files in extract_map.items():
        if not (os.path.exists(zippath) and os.path.getsize(zippath) > 1_000_000):
            print(f"  SKIP: {zippath} not downloaded")
            continue
        with zipfile.ZipFile(zippath) as z:
            names = z.namelist()
            for inner, dest in files.items():
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                # Try to find the file (zip may have different prefix)
                matches = [n for n in names if inner in n or n.endswith(inner.split("/")[-1])]
                if not matches:
                    print(f"  MISSING in zip: {inner}")
                    continue
                match = matches[0]
                if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                    print(f"  SKIP: {os.path.basename(dest)}")
                    continue
                with z.open(match) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                print(f"  OK: {os.path.basename(dest)}")

    # Exchange raw format (for LSTNet comparison baseline)
    download(
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
        f"{BASE}/Exchange/exchange_rate.txt.gz",
        "Exchange rate (LSTNet raw format)"
    )

    # METR-LA and PEMS-BAY HDF5 (from thuml)
    download_gdown("1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC", "/tmp/METR_LA.zip", "METR-LA pack")
    if os.path.exists("/tmp/METR_LA.zip") and os.path.getsize("/tmp/METR_LA.zip") > 1_000_000:
        with zipfile.ZipFile("/tmp/METR_LA.zip") as z:
            for name in z.namelist():
                if "metr-la" in name.lower() and name.endswith(".h5"):
                    dest = f"{BASE}/METR-LA/metr-la.h5"
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    if not (os.path.exists(dest) and os.path.getsize(dest) > 1_000_000):
                        with z.open(name) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                        print(f"  OK: metr-la.h5")
                if "pems-bay" in name.lower() and name.endswith(".h5"):
                    dest = f"{BASE}/PEMS-BAY/pems-bay.h5"
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    if not (os.path.exists(dest) and os.path.getsize(dest) > 1_000_000):
                        with z.open(name) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                        print(f"  OK: pems-bay.h5")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TFB DATASETS (decisionintelligence TFB benchmark pack)
#    Papers: DTAF, Channel Strategy Survey
#    Datasets: Wike2000, Covid-19 (TFB 948-channel), AQShunyi, AQWan, NN5, Wind
# ─────────────────────────────────────────────────────────────────────────────
def download_tfb():
    print("\n=== TFB Datasets (decisionintelligence benchmark pack) ===")
    TMP_TFB = "/tmp/TFB_datasets.zip"
    if not (os.path.exists(TMP_TFB) and os.path.getsize(TMP_TFB) > 10_000_000):
        download_gdown("1vgpOmAygokoUt235piWKUjfwao6KwLv7", TMP_TFB, "TFB benchmark pack (363MB)")

    if not (os.path.exists(TMP_TFB) and os.path.getsize(TMP_TFB) > 10_000_000):
        print("  TFB pack not available")
        return

    extract_map = {
        "forecasting/Wike2000.csv":   f"{BASE}/Wiki/Wike2000.csv",
        "forecasting/Covid-19.csv":   f"{BASE}/COVID-19/Covid-19_TFB.csv",
        "forecasting/AQShunyi.csv":   f"{BASE}/AQShunyi/AQShunyi.csv",
        "forecasting/AQWan.csv":      f"{BASE}/AQShunyi/AQWan.csv",
        "forecasting/NN5.csv":        f"{BASE}/NN5/NN5_TFB.csv",
        "forecasting/Wind.csv":       f"{BASE}/Wind/Wind.csv",
        "forecasting/Solar.csv":      f"{BASE}/Solar/Solar_TFB.csv",
        "forecasting/PEMS04.csv":     f"{BASE}/PEMS04/PEMS04_TFB.csv",
        "forecasting/PEMS08.csv":     f"{BASE}/PEMS08/PEMS08_TFB.csv",
        "forecasting/METR-LA.csv":    f"{BASE}/METR-LA/METR-LA_TFB.csv",
        "forecasting/PEMS-BAY.csv":   f"{BASE}/PEMS-BAY/PEMS-BAY_TFB.csv",
    }
    with zipfile.ZipFile(TMP_TFB) as z:
        for inner, dest in extract_map.items():
            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                print(f"  SKIP: {os.path.basename(dest)}")
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                with z.open(inner) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                print(f"  OK: {os.path.basename(dest)}")
            except KeyError:
                print(f"  MISSING in TFB pack: {inner}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FRETS DATASETS (FreTS paper: aikunyi/FreTS)
#    Papers: FreTS (NeurIPS 2023)
#    Datasets: ECG_data.csv (140-lead ECG), covid.csv (California hospitalization)
# ─────────────────────────────────────────────────────────────────────────────
def download_frets():
    print("\n=== FreTS Datasets (aikunyi/FreTS GitHub) ===")
    base_url = "https://raw.githubusercontent.com/aikunyi/FreTS/main/dataset"
    for fname, dest_name in [("ECG_data.csv", "ECG_data.csv"), ("covid.csv", "covid_california.csv")]:
        download(
            f"{base_url}/{fname}",
            f"{BASE}/{'ECG' if fname=='ECG_data.csv' else 'COVID-19'}/{dest_name}",
            f"FreTS {fname}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. M4 COMPETITION DATA
#    Papers: FreDF (ICLR 2025), Routing/xCPD (ArXiv 2026)
# ─────────────────────────────────────────────────────────────────────────────
def download_m4():
    print("\n=== M4 Competition Datasets (Mcompetitions/M4-methods) ===")
    base = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset"
    files = [
        ("Train/Monthly-train.csv",   "Monthly-Train.csv"),
        ("Test/Monthly-test.csv",     "Monthly-Test.csv"),
        ("Train/Quarterly-train.csv", "Quarterly-Train.csv"),
        ("Test/Quarterly-test.csv",   "Quarterly-Test.csv"),
        ("Train/Yearly-train.csv",    "Yearly-Train.csv"),
        ("Test/Yearly-test.csv",      "Yearly-Test.csv"),
        ("Train/Daily-train.csv",     "Daily-Train.csv"),
        ("Test/Daily-test.csv",       "Daily-Test.csv"),
        ("Train/Hourly-train.csv",    "Hourly-Train.csv"),
        ("Test/Hourly-test.csv",      "Hourly-Test.csv"),
        ("Train/Weekly-train.csv",    "Weekly-Train.csv"),
        ("Test/Weekly-test.csv",      "Weekly-Test.csv"),
    ]
    os.makedirs(f"{BASE}/M4", exist_ok=True)
    for remote, local in files:
        download(f"{base}/{remote}", f"{BASE}/M4/{local}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. NN5 COMPETITION DATA (Monash forecasting archive)
#    Papers: FreDF (ICLR 2025), DTAF (ArXiv 2025)
# ─────────────────────────────────────────────────────────────────────────────
def download_nn5():
    print("\n=== NN5 Datasets (Monash Time Series Forecasting Archive) ===")
    base = "https://zenodo.org/record/4656117/files"
    for fname in ["nn5_daily_dataset_with_missing_values.tsf", "nn5_weekly_dataset.tsf"]:
        download(f"{base}/{fname}?download=1", f"{BASE}/NN5/{fname}", fname)


# ─────────────────────────────────────────────────────────────────────────────
# 6. S&P500 STOCK DATA (via yfinance)
#    Papers: TimeBridge (ICML 2025) - financial forecasting experiments
#    Note: CSI500 requires manual download (see NOTES below)
# ─────────────────────────────────────────────────────────────────────────────
def download_sp500():
    print("\n=== S&P500 Stock Data (yfinance) ===")
    dest = f"{BASE}/SP500/SP500.csv"
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  SKIP: SP500.csv exists ({os.path.getsize(dest):,} bytes)")
        return

    try:
        import yfinance as yf
        import urllib.request, re
    except ImportError:
        print("  FAIL: yfinance not installed. Run: pip install yfinance")
        return

    # Get S&P500 tickers from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode('utf-8')
    tickers = list(dict.fromkeys(
        re.findall(r'class="external text"[^>]+>([A-Z][A-Z0-9.]{0,5})</a>', html)
    ))
    print(f"  Downloading {len(tickers)} S&P500 stocks (2018-2023)...")

    data = yf.download(
        tickers=tickers,
        start="2018-01-01",
        end="2023-12-31",
        progress=True,
        auto_adjust=True,
        threads=True,
    )

    # Extract adjusted close prices
    if hasattr(data.columns, 'levels'):
        close = data.xs('Close', axis=1, level=1)
    else:
        close = data[['Close']]

    # Keep only stocks with < 10% missing data
    valid = close.columns[close.isnull().mean() < 0.1].tolist()
    close = close[valid].ffill().reset_index()
    close.rename(columns={'Date': 'date'}, inplace=True)

    os.makedirs(f"{BASE}/SP500", exist_ok=True)
    close.to_csv(dest, index=False)
    print(f"  OK: SP500.csv ({os.path.getsize(dest):,} bytes, {len(close)} rows, {len(valid)} stocks)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Dataset directory: {BASE}")
    os.makedirs(BASE, exist_ok=True)

    download_standard()
    download_tfb()
    download_frets()
    download_m4()
    download_nn5()
    download_sp500()

    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED:")
    print("="*60)
    print("""
1. CSI500 (TimeBridge financial experiments)
   - 502 Chinese stocks, 2018-2023, daily close prices
   - Source: https://cn.investing.com/indices/china-securities-500
   - Alternative: Use AKShare (pip install akshare) to download A-share data
     akshare.stock_zh_index_spot() for CSI500 component list
     akshare.stock_zh_a_hist() for individual stock history
   - Save to: input/CSI500/CSI500.csv (date + 502 stock columns)

2. Crypto (Time-SSM financial experiments)
   - G-Research Crypto Forecasting competition data
   - 7 major cryptocurrencies, 1-minute frequency
   - Size: (125699, 17891, 35873) train/val/test
   - Source: https://www.kaggle.com/competitions/g-research-crypto-forecasting/
   - Requires Kaggle account: kaggle competitions download g-research-crypto-forecasting
   - Save to: input/Crypto/

3. Air Convection (Time-SSM experiments)
   - 10-variable atmospheric data, 15-min frequency, 2023
   - Source: NOAA Physical Sciences Laboratory (https://www.psl.noaa.gov/)
   - Time-SSM authors scraped this; no standard download script available
   - Size: (14647, 4882, 4882) = ~24411 samples at 15min
   - NOTE: Time-SSM code repo is NOT publicly available (as of 2026-06)
   - Save to: input/AirConvection/

4. UTSD (SEMPO/Time Tracker pre-training ONLY)
   - Universal Time Series Dataset (83M+ data points, 7 domains)
   - Only needed to pre-train SEMPO/Timer from scratch
   - For fine-tuning/evaluation: use pre-trained checkpoints in repo
   - NumPy format: https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
   - HuggingFace: pip install datasets; datasets.load_dataset("thuml/UTSD")
   - Save to: input/UTSD/utsd.npy

5. BOOM (This Time is Different / TOTO paper)
   - 350M observations, 2807 real-world observability time series
   - Source: https://huggingface.co/datasets/Datadog/BOOM
   - Download: python -c "from datasets import load_dataset; d=load_dataset('Datadog/BOOM'); d.save_to_disk('input/BOOM')"
   - NOTE: Very large dataset (100M+ observations)

6. Stock-1390 (Routing/xCPD paper)
   - 1,390 stock price series from Chen et al., 2024
   - Code and data NOT yet released by xCPD authors (arXiv 2603.13702)
   - Check: https://github.com/Clearloveyuan/xCPD for future release
""")
