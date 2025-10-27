"""
Monthly 20/40/60 Ensemble Pipeline (baseline, plug-and-play for Temporal-GAT)
-----------------------------------------------------------------------------
- Windows: T in {20, 40, 60}; labels: 2 (>=+10%), 1 (+5%~<+10%), 0 (else)
- Train on each month-end (asof). Save model per T with timestamp.
- Later, load saved models for that same asof date and produce next-month-end
  predictions, exporting a per-symbol matrix to Excel.

NOTE: This baseline flattens the [T, F] window for each symbol to a vector
[T*F] and trains a simple classifier (LogisticRegression). Replace the model
factory `make_model()` with your Temporal-GAT wrapper to use graph features.

Input file: /mnt/data/price_volume_timeseries.xlsx
Required columns: date, symbol, close, volume

CLI examples:
1) Train (for 2025-09-30 month-end):
   python temporal_gat_monthly_ensemble_pipeline.py --mode train --asof 2025-09-30

2) Predict (load the just-trained models for 2025-09-30 and export Excel):
   python temporal_gat_monthly_ensemble_pipeline.py --mode predict --asof 2025-09-30

Outputs:
- Models: /mnt/data/models/model_T{T}_{YYYYMMDD}.joblib
- Predictions: /mnt/data/pred_matrix_{YYYYMMDD}.xlsx (symbols × predictions)
"""
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------- Config --------------------------- #
# 원본 엑셀 파일 (행렬 형태)
DATA_PATH = "price_volume_timeseries.xlsx"
# 전처리된 엑셀 파일 (긴 형태, 선택적 사용)
PREPROCESSED_DATA_PATH = "preprocessed_data.xlsx"

MODEL_DIR = "models"
PRED_DIR = "output"
WINDOWS = [20, 40, 60]

# ------------------------ Utilities -------------------------- #

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)


def parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.to_period("M").to_timestamp("M")


def get_month_ends(dts: List[pd.Timestamp]) -> List[pd.Timestamp]:
    me = sorted(set([month_end(d) for d in dts]))
    return me


def next_month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthEnd(1)).normalize()


# --------------------- Feature Engineering ------------------- #
FEATURES = ["log_return", "vol_z", "volatility20", "mom20"]


def load_and_engineer(data_path: str) -> pd.DataFrame:
    """엑셀 파일을 읽어서 전처리된 데이터를 반환
    
    지원 형식:
    1. 전처리된 긴 형태 (preprocessed_data.xlsx의 'data' 시트)
    2. 행렬 형태 (price, volume 시트를 포함한 원본 엑셀)
    """
    try:
        # 먼저 전처리된 파일인지 확인 (data 시트 존재 여부)
        xl = pd.ExcelFile(data_path)
        if "data" in xl.sheet_names:
            print(f"📂 전처리된 엑셀 파일 로드: {data_path}")
            df = pd.read_excel(data_path, sheet_name="data")
            # Required columns: date, symbol, close, volume
            req = {"date", "symbol", "close", "volume"}
            missing = req - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
            # 이미 특성이 계산되어 있다면 그대로 사용
            if all(col in df.columns for col in FEATURES):
                print("   ✅ 이미 전처리된 특성 포함")
                return df
        
        # 전처리된 파일이 아니면 원본 형식 처리
        print(f"📂 원본 엑셀 파일 처리 중: {data_path}")
        
        # 시트 읽기
        df_price = pd.read_excel(data_path, sheet_name="price")
        df_volume = pd.read_excel(data_path, sheet_name="volume")
        
        # 첫 번째 컬럼이 날짜
        date_col = df_price.columns[0]
        
        # 날짜 컬럼 변환
        df_price.rename(columns={date_col: "date"}, inplace=True)
        df_price["date"] = pd.to_datetime(df_price["date"]).dt.normalize()
        
        df_volume.rename(columns={df_volume.columns[0]: "date"}, inplace=True)
        df_volume["date"] = pd.to_datetime(df_volume["date"]).dt.normalize()
        
        # Melt: [날짜 X 종목코드] → [date, symbol, value]
        df_price_long = df_price.melt(
            id_vars=["date"],
            var_name="symbol",
            value_name="close"
        )
        
        df_volume_long = df_volume.melt(
            id_vars=["date"],
            var_name="symbol",
            value_name="volume"
        )
        
        # 두 데이터프레임 합치기
        df = pd.merge(
            df_price_long,
            df_volume_long,
            on=["date", "symbol"],
            how="inner"
        )
        
        # Required columns: date, symbol, close, volume
        req = {"date", "symbol", "close", "volume"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Failed to load data from {data_path}: {e}")

    # log return
    df["log_return"] = (
        np.log(df.groupby("symbol")["close"].pct_change().add(1.0))
    )

    # log(volume) and z-score over 60-day window by symbol
    df["log_vol"] = np.log(df["volume"].fillna(0) + 1)
    win = 60
    grp = df.groupby("symbol")
    df["vol_z"] = grp["log_vol"].transform(
        lambda s: (s - s.rolling(win).mean()) / (s.rolling(win).std() + 1e-8)
    )

    # rolling volatility (20d std of returns)
    df["volatility20"] = grp["log_return"].transform(lambda s: s.rolling(20).std())

    # mom20 (cumulative log return over 20d)
    df["mom20"] = grp["log_return"].transform(lambda s: s.rolling(20).sum())

    # NaN 값 처리 및 2021-01-26 이후 데이터만 사용
    print(f"   [INFO] NaN 값 처리 전: {len(df)} 행")
    # 특성 컬럼(파생 지표)에 NaN이 있는 행 제거
    df = df.dropna(subset=FEATURES)
    print(f"   [INFO] NaN 제거 후: {len(df)} 행")
    
    # 2021-01-26 이후 데이터만 사용
    cutoff_date = pd.Timestamp("2021-01-26")
    df = df[df["date"] >= cutoff_date].copy()
    print(f"   [INFO] 2021-01-26 이후: {len(df)} 행")
    print(f"   [INFO] 날짜 범위: {df['date'].min()} ~ {df['date'].max()}")

    return df


# --------------------- Dataset Construction ------------------ #
@dataclass
class WindowedSample:
    X: np.ndarray  # shape [N, T*F] for a single asof
    y: np.ndarray  # shape [N] labels for that asof (if available)
    symbols: List[str]


def build_window_matrix(
    df: pd.DataFrame,
    symbols: List[str],
    asof: pd.Timestamp,
    T: int,
    features: List[str] = FEATURES,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build [N, T*F] matrix for a single asof date.
    Returns (X, mask, symbols_kept) where mask marks rows that are fully valid.
    """
    start = asof - pd.tseries.offsets.BDay(T - 1)
    # filter window rows
    wdf = df[(df["date"] >= start) & (df["date"] <= asof)][["date", "symbol"] + features]
    # pivot to panel-like: index=(symbol,date), columns=features
    # we'll later order by date ascending and flatten
    wdates = pd.bdate_range(start, asof)

    X_list = []
    mask = []
    kept = []
    for sym in symbols:
        sdf = (
            wdf[wdf["symbol"] == sym]
            .set_index("date")[features]
            .reindex(wdates)
        )
        if sdf.isna().values.any():
            mask.append(False)
            X_list.append(np.zeros((T, len(features)), dtype=np.float32))
            continue
        kept.append(sym)
        mask.append(True)
        X_list.append(sdf.values.astype(np.float32))

    X = np.stack(X_list, axis=0)  # [N, T, F]
    X = X.reshape(X.shape[0], -1)  # [N, T*F]
    return X, np.array(mask, dtype=bool), kept


def build_label_vector(
    df: pd.DataFrame,
    symbols: List[str],
    asof: pd.Timestamp,
) -> np.ndarray:
    """Label is next-month-end return bucket relative to asof price.
       2: >= +10%, 1: +5% ~ < +10%, 0: else
    """
    next_me = next_month_end(asof)

    base = (
        df[df["date"] == asof][["symbol", "close"]]
        .set_index("symbol")
        .reindex(symbols)["close"]
    )
    fut = (
        df[df["date"] == next_me][["symbol", "close"]]
        .set_index("symbol")
        .reindex(symbols)["close"]
    )
    rel = (fut / base - 1.0).astype(float)
    y = np.zeros(len(symbols), dtype=np.int64)
    y[(rel >= 0.05) & (rel < 0.10)] = 1
    y[rel >= 0.10] = 2
    # NaNs (no price on base or future) remain 0 by default; could mask if desired
    return y


# ------------------------ Model Factory ---------------------- #

def make_model() -> Pipeline:
    """Return a scikit-learn pipeline. Replace with your Temporal-GAT wrapper.
    For Temporal-GAT, create a class with fit(X,y) and predict_proba(X).
    """
    clf = LogisticRegression(max_iter=2000, multi_class="ovr", n_jobs=None)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])
    return pipe


# -------------------------- Train ---------------------------- #

def train_month_end_models(df: pd.DataFrame, asof: pd.Timestamp) -> Dict[int, str]:
    """Train one model per window T using all month-end samples up to `asof`.
    Returns mapping T -> model_path saved.
    """
    ensure_dirs()
    all_dates = sorted(df["date"].unique())
    if asof not in all_dates:
        raise ValueError(f"asof {asof.date()} not found in data dates.")

    # restrict to month-ends up to asof - we predict next month from asof
    month_ends = [d for d in get_month_ends(all_dates) if d <= asof]

    symbols = sorted(df["symbol"].unique().tolist())
    saved = {}

    for T in WINDOWS:
        # 80 business days policy check
        min_span = pd.bdate_range(asof - pd.tseries.offsets.BDay(79), asof)
        if len(min_span) < 80:
            raise ValueError("Need at least 80 business days before asof.")

        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []

        for me in month_ends:
            # require we can also label (need next month end)
            nme = next_month_end(me)
            if nme not in all_dates:
                continue

            X_me, mask_me, kept_syms = build_window_matrix(df, symbols, me, T)
            if not mask_me.any():
                continue
            y_me_full = build_label_vector(df, symbols, me)
            y_me = y_me_full[mask_me]
            X_me = X_me[mask_me]

            X_all.append(X_me)
            y_all.append(y_me)

        if not X_all:
            raise ValueError(f"No training samples for T={T} up to {asof.date()}.")

        Xtr = np.vstack(X_all)
        ytr = np.concatenate(y_all)

        model = make_model()
        model.fit(Xtr, ytr)
        model_path = os.path.join(MODEL_DIR, f"model_T{T}_{asof.strftime('%Y%m%d')}.joblib")
        joblib.dump({"model": model, "features": FEATURES, "T": T, "asof": asof}, model_path)
        saved[T] = model_path
        print(f"Saved: {model_path}  (X:{Xtr.shape}, y:{ytr.shape})")

    return saved


# ------------------------- Predict --------------------------- #

def predict_next_month(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Load models for the given asof, predict per T and ensemble, then export Excel."""
    ensure_dirs()
    symbols = sorted(df["symbol"].unique().tolist())

    results: Dict[int, Dict[str, np.ndarray]] = {}

    for T in WINDOWS:
        model_path = os.path.join(MODEL_DIR, f"model_T{T}_{asof.strftime('%Y%m%d')}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Train first.")
        pack = joblib.load(model_path)
        model: Pipeline = pack["model"]

        X, mask, kept = build_window_matrix(df, symbols, asof, T)
        # Predict only for valid rows
        proba = np.full((len(symbols), 3), np.nan, dtype=float)
        if mask.any():
            p = model.predict_proba(X[mask])
            # Ensure columns order is [0,1,2]; LogisticRegression does that
            proba[mask] = p
        results[T] = {"proba": proba, "mask": mask}

    # Ensemble: average probabilities across Ts where available
    P_stack = []
    valid_mask = np.zeros(len(symbols), dtype=int)
    for T in WINDOWS:
        pm = results[T]["proba"]
        m = results[T]["mask"]
        # Replace NaN rows with zeros to not influence avg; track counts
        P_stack.append(np.nan_to_num(pm, nan=0.0))
        valid_mask += m.astype(int)

    P_sum = np.sum(P_stack, axis=0)  # [N,3]
    # Avoid div by zero; where valid_mask==0 leave NaNs
    P_avg = np.full_like(P_sum, np.nan, dtype=float)
    for i, cnt in enumerate(valid_mask):
        if cnt > 0:
            P_avg[i] = P_sum[i] / cnt

    pred_ens = np.nanargmax(P_avg, axis=1)

    # Build per-T discrete preds as well
    perT_preds = {}
    for T in WINDOWS:
        pm = results[T]["proba"]
        predT = np.full(len(symbols), np.nan)
        m = results[T]["mask"]
        if m.any():
            predT[m] = np.argmax(pm[m], axis=1)
        perT_preds[T] = predT

    # Assemble output DataFrame
    out = pd.DataFrame({"symbol": symbols})
    # Ensemble probs
    out["proba_ens_0"] = P_avg[:, 0]
    out["proba_ens_1"] = P_avg[:, 1]
    out["proba_ens_2"] = P_avg[:, 2]
    out["pred_ensemble"] = pred_ens

    # Per-T probs and preds
    for T in WINDOWS:
        pm = results[T]["proba"]
        out[f"proba_T{T}_0"] = pm[:, 0]
        out[f"proba_T{T}_1"] = pm[:, 1]
        out[f"proba_T{T}_2"] = pm[:, 2]
        out[f"pred_T{T}"] = perT_preds[T]

    # Save Excel
    xlsx_path = os.path.join(PRED_DIR, f"pred_matrix_{asof.strftime('%Y%m%d')}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as wr:
        out.to_excel(wr, index=False, sheet_name="predictions")

    print(f"Exported predictions to: {xlsx_path}")
    return out


# --------------------------- Main ---------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--asof", type=str, required=True, help="Month-end date (YYYY-MM-DD)")
    parser.add_argument("--data", type=str, default=None, help="데이터 파일 경로 (기본: DATA_PATH)")
    args = parser.parse_args()

    asof = parse_date(args.asof)
    asof = month_end(asof)

    # 데이터 파일 경로 결정
    data_path = args.data or DATA_PATH
    # 전처리된 파일이 있으면 우선 사용
    if not args.data and os.path.exists(PREPROCESSED_DATA_PATH):
        data_path = PREPROCESSED_DATA_PATH
    
    df = load_and_engineer(data_path)

    # Safety check: enforce 80 business days before asof
    if len(pd.bdate_range(asof - pd.tseries.offsets.BDay(79), asof)) < 80:
        raise SystemExit("ERROR: Need at least 80 business days of data prior to asof.")

    if args.mode == "train":
        train_month_end_models(df, asof)
    else:
        predict_next_month(df, asof)


if __name__ == "__main__":
    main()
