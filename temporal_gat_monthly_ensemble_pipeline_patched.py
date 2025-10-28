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
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# === Patched: Use Excel trading calendar (actual KRX trading days) ===
TRADING_CAL: pd.DatetimeIndex | None = None

def set_trading_calendar_from_df(df: pd.DataFrame) -> None:
    """df['date']로부터 전역 TRADING_CAL(실제 거래일 캘린더) 생성"""
    global TRADING_CAL

    # 1) 가장 견고한 변환: 모두 UTC로 파싱 → 로컬 naive로 변환 → 자정 정규화
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True) \
                .dt.tz_convert(None) \
                .dt.normalize()

    # 2) (선택) 엑셀 일련번호(예: 44561)만 섞인 특수 케이스까지 잡고 싶으면:
    if dates.isna().mean() > 0.8 and pd.api.types.is_numeric_dtype(df["date"]):
        dates = pd.to_datetime(df["date"].astype("float64"),
                               unit="d", origin="1899-12-30",
                               errors="coerce").dt.normalize()

    cal = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    TRADING_CAL = cal

def snap_to_calendar(asof: pd.Timestamp, cal: pd.DatetimeIndex) -> pd.Timestamp | None:
    """If asof not in calendar, snap to the previous trading day."""
    if cal is None or len(cal) == 0:
        return None
    asof = pd.Timestamp(asof).tz_localize(None).normalize()
    pos = cal.searchsorted(asof, side="right") - 1
    if pos < 0:
        return None
    return cal[pos]

def last_n_trading_days(asof: pd.Timestamp, n: int, cal: pd.DatetimeIndex) -> pd.DatetimeIndex | None:
    if cal is None or len(cal) == 0:
        return None
    a = snap_to_calendar(asof, cal)
    if a is None:
        return None
    pos = cal.get_indexer([a])[0]
    start = pos - (n - 1)
    if start < 0:
        return None
    return cal[start:pos+1]
# === End Patch Header ===

# Numba for JIT compilation
NUMBA_AVAILABLE = False
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    pass  # Numba is optional

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


def get_month_ends(cal: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if cal is None or len(cal) == 0:
        return []
    per = cal.to_period("M")
    s = pd.Series(cal, index=per)
    month_ends = s.groupby(level=0).max()
    month_ends = month_ends[month_ends >= pd.Timestamp("2021-04-30")]
    return month_ends.tolist()


def next_month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthEnd(1)).normalize()


def assert_trading_cal_ready(df: pd.DataFrame) -> None:
    """Ensure global TRADING_CAL is initialized from df['date']."""
    global TRADING_CAL
    if TRADING_CAL is None or len(TRADING_CAL) == 0:
        set_trading_calendar_from_df(df)
        print(f"[INFO] Trading calendar initialized: "
              f"{TRADING_CAL[0].date()} ~ {TRADING_CAL[-1].date()} "
              f"({len(TRADING_CAL)} days)")


def _ensure_datetime_series(s: pd.Series) -> pd.Series:
    """엑셀 date 컬럼을 확실히 datetime으로 변환 (문자열/일련번호 모두 지원)"""
    s1 = pd.to_datetime(s, errors="coerce")  # 문자열/타임스탬프 케이스
    # 만약 대부분 NaT면 엑셀 일련번호로 가정
    if s1.isna().mean() > 0.8 and np.issubdtype(s.dtype, np.number):
        s1 = pd.to_datetime(s.astype("float64"), unit="d",
                            origin="1899-12-30", errors="coerce")
    return s1.dt.tz_localize(None).dt.normalize()


# --------------------- Feature Engineering ------------------- #
FEATURES = ["log_return", "vol_z", "volatility20", "mom20"]


def load_and_engineer(data_path: str) -> pd.DataFrame:
    """엑셀 파일을 읽어서 전처리된 데이터를 반환

    지원 형식:
    1) 전처리된 긴 형태 (preprocessed_data.xlsx의 'data' 시트)
    2) 행렬 형태 (price, volume 시트를 포함한 원본 엑셀)
    """
    try:
        xl = pd.ExcelFile(data_path)
        # 1) 이미 긴 형태로 전처리된 파일이면 그대로 로드
        if "data" in xl.sheet_names:
            print(f"전처리된 엑셀 파일 로드: {data_path}")
            df = pd.read_excel(data_path, sheet_name="data")
            # df["date"] = _ensure_datetime_series(df["date"])

            # 빈 셀을 0으로 채움
            df["close"] = df["close"].fillna(0)
            df["volume"] = df["volume"].fillna(0)
            
            req = {"date", "symbol", "close", "volume"}
            missing = req - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
            # 이미 특성이 계산되어 있으면 바로 반환
            if all(col in df.columns for col in FEATURES):
                print("   이미 전처리된 특성 포함")
                return df
        
        # 2) 'data' 시트가 없거나 특성이 없으면 → 원본 형식 처리
        # ('data' 시트가 있는데 특성이 없는 경우도 여기로 옴)
        print(f"원본 엑셀 파일 처리 중: {data_path}")
        df_price = pd.read_excel(data_path, sheet_name="price").fillna(0)
        # df_price["date"] = _ensure_datetime_series(df_price["date"])
        df_volume = pd.read_excel(data_path, sheet_name="volume").fillna(0)
        # df_volume["date"] = _ensure_datetime_series(df_volume["date"])

        # 날짜 컬럼 명 통일
        date_col = df_price.columns[0]
        df_price = df_price.rename(columns={date_col: "date"})
        df_price["date"] = pd.to_datetime(df_price["date"]).dt.normalize()

        df_volume = df_volume.rename(columns={df_volume.columns[0]: "date"})
        df_volume["date"] = pd.to_datetime(df_volume["date"]).dt.normalize()

        # Wide → Long
        df_price_long = df_price.melt(id_vars=["date"], var_name="symbol", value_name="close")
        df_volume_long = df_volume.melt(id_vars=["date"], var_name="symbol", value_name="volume")

        # 머지 및 기본 검사
        df = pd.merge(df_price_long, df_volume_long, on=["date", "symbol"], how="inner")
        
        req = {"date", "symbol", "close", "volume"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    except Exception as e:
        raise ValueError(f"Failed to load data from {data_path}: {e}")
    
    # Feature engineering
    # log return
    df["log_return"] = (
        np.log(df.groupby("symbol")["close"].pct_change().add(1.0))
    )
    print(f"   [DEBUG] Feature engineering 후 data shape: {df.shape}")
    print(f"   [DEBUG] log_return NaN: {df['log_return'].isna().sum()} rows")

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

    # NaN 값 처리 및 2021-04-30 이후 데이터만 사용
    print(f"   [INFO] NaN 값 처리 전: {len(df)} 행")
    # 특성 컬럼(파생 지표)에 NaN이 있는 행 제거
    df = df.dropna(subset=FEATURES)
    print(f"   [INFO] NaN 제거 후: {len(df)} 행")
    
    # 2021-04-30 이후 데이터만 사용
    cutoff_date = pd.Timestamp("2021-04-30")
    df = df[df["date"] >= cutoff_date].copy()
    print(f"   [INFO] 2021-04-30 이후: {len(df)} 행")
    print(f"   [INFO] 날짜 범위: {df['date'].min()} ~ {df['date'].max()}")

    return df


# --------------------- Dataset Construction ------------------ #
@dataclass
class WindowedSample:
    X: np.ndarray  # shape [N, T*F] for a single asof
    y: np.ndarray  # shape [N] labels for that asof (if available)
    symbols: List[str]


def group_data_by_symbol(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """전처리된 데이터를 종목별로 분리하여 딕셔너리로 반환

    Args:
        df: (date, symbol)을 PK로 하는 긴 형태 데이터프레임

    Returns:
        {symbol: DataFrame} 형태의 딕셔너리
    """
    symbol_groups = {}
    for symbol in df["symbol"].unique():
        symbol_df = df[df["symbol"] == symbol].copy()
        # 보정 추가
        # symbol_df["date"] = _ensure_datetime_series(symbol_df["date"])
        symbol_df = symbol_df.set_index("date").sort_index()

        # (선택) 강제 캐스팅으로 확실히 DatetimeIndex
        symbol_df.index = pd.DatetimeIndex(symbol_df.index)

        symbol_groups[symbol] = symbol_df
    return symbol_groups


def extract_window_for_symbol(
    symbol_df: pd.DataFrame,
    asof: pd.Timestamp,
    T: int,
    features: List[str] = FEATURES,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """특정 종목에 대해 윈도우 데이터 추출
    
    Args:
        symbol_df: 해당 종목의 데이터프레임 (date가 인덱스)
        asof: 기준일
        T: 윈도우 크기 (영업일 수)
        features: 사용할 특성 리스트
        debug: 디버그 모드
    
    Returns:
        (X, returns, is_valid)
        - X: [T, F] 형태의 윈도우 데이터
        - returns: [T] 형태의 수익률 데이터
        - is_valid: 유효한 데이터인지 여부
    """
    global TRADING_CAL
    wdates = last_n_trading_days(asof, T, TRADING_CAL)
    if wdates is None or len(wdates) != T:
        if debug:
            print(f"    [DEBUG] 캘린더 부족: asof={asof.date()}, T={T}")
        return np.zeros((T, len(features)), np.float32), np.zeros((T,), np.float32), False

    if debug:
        w_start = wdates[0]
        w_end = wdates[-1]
        print(f"    [DEBUG] 윈도우 기간: {w_start.date()} ~ {w_end.date()}")
        smin = symbol_df.index.min()
        smax = symbol_df.index.max()
        try:
            print(f"    [DEBUG] 종목 데이터 date range: {smin.date()} ~ {smax.date()}")
        except Exception:
            print(f"    [DEBUG] 종목 데이터 date range: {smin} ~ {smax}")
        print(f"    [DEBUG] 필요한 날짜 수: {len(wdates)}, 종목 데이터 행 수: {len(symbol_df)}")

    # 윈도우 데이터 추출
    # Robust selection (index-agnostic): 'date'로 머지해서 정렬
    win_df = pd.DataFrame({'date': pd.DatetimeIndex(wdates)})
    sym_reset = symbol_df.reset_index()  # 'date'를 컬럼으로
    cols = ['date'] + list(features)
    sym_reset = sym_reset[cols]
    selected_df = win_df.merge(sym_reset, on='date', how='left').set_index('date')

    # NaN 체크
    nan_rows = selected_df.isna().any(axis=1).sum()
    if nan_rows > 0:
        if debug:
            print(f"    [DEBUG] NaN 있는 행 수: {nan_rows}/{len(selected_df)}")
        return np.zeros((T, len(features)), dtype=np.float32), np.zeros((T,), dtype=np.float32), False
    
    X = selected_df.values.astype(np.float32)
    
    # returns 추출
    if "log_return" in features:
        returns = selected_df["log_return"].values.astype(np.float32)
    else:
        returns = selected_df.iloc[:, 0].values.astype(np.float32)
    
    if debug:
        print(f"    [DEBUG] 유효 데이터 추출 성공!")
    
    return X, returns, True


def build_window_matrix(
    df: pd.DataFrame,
    symbols: List[str],
    asof: pd.Timestamp,
    T: int,
    features: List[str] = FEATURES,
    symbol_groups: Dict[str, pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build [N, T*F] matrix for a single asof date.
    Returns (X, mask, symbols_kept) where mask marks rows that are fully valid.
    
    Also extracts returns separately for correlation computation.
    
    Args:
        df: 전체 데이터프레임
        symbols: 종목 리스트
        asof: 기준일
        T: 윈도우 크기
        features: 사용할 특성 리스트
        symbol_groups: 종목별로 미리 그룹화된 데이터 (선택적, 캐시로 사용)
    """
    # 종목별 그룹 데이터 생성 (캐시되지 않은 경우)
    if symbol_groups is None:
        symbol_groups = group_data_by_symbol(df)
    
    X_list = []
    returns_list = []  # 수익률 데이터 분리
    mask = []
    kept = []
    
    for sym in symbols:
        if sym not in symbol_groups:
            mask.append(False)
            X_list.append(np.zeros((T, len(features)), dtype=np.float32))
            returns_list.append(np.zeros((T,), dtype=np.float32))
            continue
        
        symbol_df = symbol_groups[sym]
        # 첫 번째 종목만 디버그 출력
        debug_mode = (len(mask) == 0)
        X_sym, returns_sym, is_valid = extract_window_for_symbol(symbol_df, asof, T, features, debug=debug_mode)
        
        if not is_valid:
            mask.append(False)
            X_list.append(np.zeros((T, len(features)), dtype=np.float32))
            returns_list.append(np.zeros((T,), dtype=np.float32))
            continue
        
        kept.append(sym)
        mask.append(True)
        X_list.append(X_sym)
        returns_list.append(returns_sym)

    X = np.stack(X_list, axis=0)  # [N, T, F]
    X = X.reshape(X.shape[0], -1)  # [N, T*F]
    
    # 수익률 데이터도 반환
    returns_array = np.stack(returns_list, axis=0)  # [N, T]
    
    return X, np.array(mask, dtype=bool), kept, returns_array


def build_label_vector(
    df: pd.DataFrame,
    symbols: List[str],
    asof: pd.Timestamp,
) -> np.ndarray:
    """Label is next 20-business-day return bucket relative to asof price.
       2: >= +10%, 1: +5% ~ < +10%, 0: else
    """
    # asof 기준 20영업일 후 날짜
    future_date = asof + pd.tseries.offsets.BDay(20)

    base = (
        df[df["date"] == asof][["symbol", "close"]]
        .set_index("symbol")
        .reindex(symbols)["close"]
    )
    fut = (
        df[df["date"] == future_date][["symbol", "close"]]
        .set_index("symbol")
        .reindex(symbols)["close"]
    )
    rel = (fut / base - 1.0).astype(float)
    y = np.zeros(len(symbols), dtype=np.int64)
    y[(rel >= 0.05) & (rel < 0.10)] = 1
    y[rel >= 0.10] = 2
    # NaNs (no price on base or future) remain 0 by default; could mask if desired
    return y


# ------------------------ Temporal-GAT + GRU Model ---------------------- #

class TemporalGATGRU(nn.Module):
    """Temporal Graph Attention Network + GRU for time series prediction"""
    
    def __init__(self, num_features, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        # GAT layers
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.2)
        
        # GRU layers
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Output layers
        self.fc = nn.Linear(hidden_dim, 3)  # 3 classes
        
    def forward(self, x, edge_index, batch):
        """Forward pass"""
        # GAT processing
        import torch.nn.functional as F_func
        x = F_func.relu(self.gat1(x, edge_index))
        x = F_func.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        
        # Pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Reshape for GRU (adding time dimension)
        # Assuming x is [batch, hidden_dim], add time dim [batch, 1, hidden_dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # GRU processing
        x, _ = self.gru(x)
        
        # Final prediction
        x = self.fc(x[:, -1, :])
        return x


class TemporalGATWrapper:
    """Wrapper for Temporal-GAT + GRU model that implements sklearn interface"""
    
    def __init__(self, T=20, hidden_dim=64, num_layers=2, num_heads=4, lr=0.001, epochs=50, corr_threshold=0.3):
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.lr = lr
        self.epochs = epochs
        self.corr_threshold = corr_threshold  # 상관계수 threshold
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.correlation_matrix = None  # 종목 간 상관관계 행렬
        
    @staticmethod
    def _compute_cosine_similarity_numba(series):
        """Numba-accelerated cosine similarity (if numba available)"""
        if NUMBA_AVAILABLE:
            @jit(nopython=True, parallel=True, nogil=True)
            def _compute_numba(series):
                n = series.shape[0]
                norms = np.empty((n,), dtype=np.float32)
                for i in prange(n):
                    norms[i] = np.sqrt(np.sum(series[i] ** 2))
                similarity = np.abs(series @ series.T / (norms[:, None] * norms[None, :] + 1e-8))
                return similarity
            return _compute_numba(series)
        else:
            # Fallback numpy
            norms = np.linalg.norm(series, axis=1, keepdims=True)
            series_norm = series / (norms + 1e-8)
            return np.abs(series_norm @ series_norm.T)
    
    @staticmethod  
    def _compute_correlation_fast_numba(returns):
        """Numba-accelerated Pearson correlation calculation"""
        if NUMBA_AVAILABLE:
            @jit(nopython=True, parallel=True, nogil=True)
            def _corrcoef_fast(data):
                """Fast correlation matrix calculation using Numba"""
                n_stocks = data.shape[0]
                n_time = data.shape[1]
                corr_matrix = np.zeros((n_stocks, n_stocks), dtype=np.float32)
                
                # Compute mean for each stock
                means = np.zeros(n_stocks, dtype=np.float32)
                for i in prange(n_stocks):
                    means[i] = np.mean(data[i])
                
                # Compute correlation matrix
                for i in prange(n_stocks):
                    for j in range(n_stocks):
                        if i == j:
                            corr_matrix[i, j] = 1.0
                        else:
                            # Compute covariance
                            cov = 0.0
                            var_i = 0.0
                            var_j = 0.0
                            for t in range(n_time):
                                diff_i = data[i, t] - means[i]
                                diff_j = data[j, t] - means[j]
                                cov += diff_i * diff_j
                                var_i += diff_i * diff_i
                                var_j += diff_j * diff_j
                            
                            # Correlation
                            denom = np.sqrt(var_i * var_j)
                            if denom > 1e-8:
                                corr_matrix[i, j] = np.abs(cov / denom)
                            else:
                                corr_matrix[i, j] = 0.0
                
                return corr_matrix
            
            return _corrcoef_fast(returns)
        else:
            # Fallback to numpy
            correlation_matrix = np.corrcoef(returns)
            return np.abs(correlation_matrix)
    
    def _compute_correlation_matrix(self, X_batch, return_data=None):
        """Compute correlation matrix between stocks using returns only
        
        Args:
            X_batch: [batch_size=종목수, T, F]
            return_data: [batch_size, T] 수익률 데이터 (log_return만)
        
        Returns:
            adjacency matrix based on returns correlation
        """
        if return_data is not None:
            # 수익률 데이터만 사용
            # return_data: [batch_size, T]
            if return_data.is_cuda:
                returns_np = return_data.cpu().numpy()
            else:
                returns_np = return_data.numpy()
            
            # 종목 간 수익률 상관관계 계산 - Numba 가속
            if NUMBA_AVAILABLE:
                correlation_matrix = self._compute_correlation_fast_numba(returns_np.astype(np.float32))
            else:
                correlation_matrix = np.corrcoef(returns_np)
                correlation_matrix = np.abs(correlation_matrix)  # absolute correlation
            
        else:
            # 기존 방식: 전체 특성 사용
            batch_size, T, F = X_batch.shape
            stock_series_np = X_batch.reshape(batch_size, -1).cpu().numpy() if X_batch.is_cuda else X_batch.reshape(batch_size, -1).numpy()
            
            if NUMBA_AVAILABLE:
                similarity_matrix = self._compute_cosine_similarity_numba(stock_series_np.astype(np.float32))
            else:
                norms = np.linalg.norm(stock_series_np, axis=1, keepdims=True)
                stock_series_norm = stock_series_np / (norms + 1e-8)
                similarity_matrix = np.abs(stock_series_norm @ stock_series_norm.T)
            
            correlation_matrix = similarity_matrix
        
        # Threshold to create binary adjacency
        adj = (correlation_matrix > self.corr_threshold).astype(np.float32)
        
        return torch.FloatTensor(adj).to(self.device)
    
    def _correlation_to_edges(self, corr_matrix):
        """Convert correlation matrix to edge_index format"""
        # corr_matrix: [num_nodes, num_nodes]
        # Convert to edge_index: [2, num_edges]
        num_nodes = corr_matrix.shape[0]
        
        # Get indices where correlation is above threshold
        rows, cols = torch.where(corr_matrix > 0)
        
        # Remove self-loops
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        
        # Create bidirectional edges (undirected graph)
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ])
        
        return edge_index
    
    def _build_graph(self, X_flat, returns_flat=None):
        """Build graph structure from correlation using returns data
        
        Args:
            X_flat: [batch_size, T*F] 전체 특성 (tensor or numpy array)
            returns_flat: [batch_size, T] 수익률 데이터 (선택적, numpy array)
        
        Returns:
            X: [batch_size, T, F] reshaped features (tensor)
            edge_index: [2, num_edges] (tensor)
        """
        # Ensure X_flat is a tensor and on the correct device
        if not isinstance(X_flat, torch.Tensor):
            X_flat = torch.FloatTensor(X_flat).to(self.device)
        else:
            X_flat = X_flat.to(self.device)
        
        # X_flat: [batch_size, T*F] -> [batch_size, T, F]
        batch_size = X_flat.shape[0]
        T, F = self.T, X_flat.shape[1] // self.T
        
        if X_flat.shape[1] % self.T != 0:
            raise ValueError(f"Input dimension {X_flat.shape[1]} is not divisible by T={self.T}")
        
        X = X_flat.reshape(batch_size, T, F)
        
        # 수익률 데이터가 제공되면 그것만 사용
        if returns_flat is not None:
            # returns_flat: [batch_size, T]
            returns_tensor = torch.FloatTensor(returns_flat).to(self.device)
            corr_matrix = self._compute_correlation_matrix(X, return_data=returns_tensor)
        else:
            # 기존 방식: 전체 특성 사용
            corr_matrix = self._compute_correlation_matrix(X)
        
        # Convert to edges
        edge_index = self._correlation_to_edges(corr_matrix)
        
        return X, edge_index
        
    def fit(self, X, y, returns_data=None, group_by_month=False):
        """Train the model
        
        Args:
            X: [batch_size, T*F] input features
            y: [batch_size] labels
            returns_data: [batch_size, T] returns data for correlation (optional)
            group_by_month: If True, process stocks by month (grouped samples)
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Model initialization
        num_features = int(X_tensor.shape[1] // self.T)
        if num_features <= 0:
            raise ValueError(f"Invalid num_features: {num_features} (shape={X_tensor.shape}, T={self.T})")
        self.model = TemporalGATGRU(num_features, self.hidden_dim, self.num_layers, self.num_heads)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        
        if group_by_month and returns_data is not None:
            # 월말 단위로 종목 묶어서 처리
            print("  [INFO] 월말 단위 종목 그룹 처리...")
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs_list = []
                y_list = []
                
                # X_scaled는 [총샘플수, T*F] 형태
                # 이를 월말별로 나누어 처리해야 함
                # 여기서는 간단히 모든 샘플을 N종목×T형태로 reshape
                
                # 전체 샘플을 한 번에 처리
                batch_size, total_features = X_tensor.shape
                N_stocks = batch_size  # 실제로는 종목 수
                T, F = self.T, total_features // self.T
                
                if N_stocks < 2:
                    # 샘플이 1개면 그래프 구성 불가능
                    raise ValueError("Need at least 2 stocks to build a correlation graph")
                
                # Reshape: [N종목, T, F]
                X_reshaped = X_tensor.reshape(N_stocks, T, F)
                
                # 수익률 데이터도 reshape
                returns_tensor = torch.FloatTensor(returns_data).to(self.device)
                
                # 상관관계 계산 (종목 간)
                returns_np = returns_tensor.cpu().numpy()
                if NUMBA_AVAILABLE:
                    corr_matrix = self._compute_correlation_fast_numba(returns_np.astype(np.float32))
                else:
                    corr_matrix = np.abs(np.corrcoef(returns_np))
                
                # 인접 행렬 생성
                adj = torch.FloatTensor(corr_matrix > self.corr_threshold).to(self.device)
                
                # edge_index 생성
                rows, cols = torch.where(adj > 0)
                mask = rows != cols
                rows = rows[mask]
                cols = cols[mask]
                edge_index = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])])
                
                # Node features: [N, F] (시간 평균)
                node_features = torch.mean(X_reshaped, dim=1)  # [N, F]
                
                # GAT 처리
                batch_indices = torch.zeros(N_stocks, dtype=torch.long, device=self.device)
                h = self.model.gat1(node_features, edge_index)
                h = torch.nn.functional.relu(h)
                h = torch.nn.functional.dropout(h, p=0.2, training=True)
                h = self.model.gat2(h, edge_index)
                
                # Pooling: [N, F] → [1, F] (전체 종목 집계)
                h_pooled = global_mean_pool(h, batch_indices)
                
                # GRU
                h = h_pooled.unsqueeze(1)
                h, _ = self.model.gru(h)
                
                # Output
                out = self.model.fc(h[:, -1, :])  # [1, 3]
                outputs_list.append(out)
                y_list.append(y_tensor[0:1])  # 첫 샘플 레이블만 사용 (필요시 수정)
                
                if outputs_list:
                    outputs = torch.cat(outputs_list, dim=0)
                    y_batch = torch.cat(y_list, dim=0)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    if (epoch + 1) % 10 == 0:
                        print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        else:
            # 기존 방식: 샘플별로 처리 (GAT 의미 없음)
            print("  [INFO] 샘플별 처리 (GAT 기능 제한적)...")
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs_list = []
                
                for i in range(X_tensor.shape[0]):
                    X_sample = X_tensor[i:i+1]
                    if returns_data is not None:
                        returns_sample = returns_data[i:i+1]
                        X_reshaped_sample, edge_index_sample = self._build_graph(X_sample, returns_sample)
                    else:
                        X_reshaped_sample, edge_index_sample = self._build_graph(X_sample)
                    
                    node_features = torch.mean(X_reshaped_sample, dim=1)
                    
                    batch_indices = torch.zeros(node_features.shape[0], dtype=torch.long, device=self.device)
                    h = self.model.gat1(node_features, edge_index_sample)
                    h = torch.nn.functional.relu(h)
                    h = torch.nn.functional.dropout(h, p=0.2, training=True)
                    h = self.model.gat2(h, edge_index_sample)
                    h = global_mean_pool(h, batch_indices)
                    h = h.unsqueeze(1)
                    h, _ = self.model.gru(h)
                    out = self.model.fc(h[:, -1, :])
                    outputs_list.append(out)
                
                outputs = torch.cat(outputs_list, dim=0)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def fit_by_month(self, X_all: List[np.ndarray], y_all: List[np.ndarray], 
                      returns_all: List[np.ndarray], month_ends: List[pd.Timestamp]):
        """월말별로 종목 묶어서 GAT 학습
        
        Args:
            X_all: 월말별 종목 데이터 리스트 [[종목수1, T*F], [종목수2, T*F], ...]
            y_all: 월말별 레이블 리스트
            returns_all: 월말별 수익률 리스트
            month_ends: 월말 날짜 리스트
        """
        print(f"  [INFO] 월말별 그래프 처리 시작 ({len(month_ends)}개 월말)")
        
        # Standardize features (모든 월말 데이터로 스케일러 학습)
        X_combined = np.vstack(X_all)
        X_scaled_combined = self.scaler.fit_transform(X_combined)
        
        # Model initialization
        T, F = self.T, X_combined.shape[1] // self.T
        self.model = TemporalGATGRU(F, self.hidden_dim, self.num_layers, self.num_heads)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs_list = []
            y_list = []
            
            # 각 월말별로 처리
            for month_end, X_month, y_month, returns_month in zip(month_ends, X_all, y_all, returns_all):
                # Standardize
                X_flat = X_month.reshape(-1, X_month.shape[-1])
                X_scaled = self.scaler.transform(X_flat)
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                N_stocks = X_tensor.shape[0]
                if N_stocks < 2:
                    continue  # 종목이 1개면 건너뜀
                
                # Reshape: [N종목, T, F]
                X_reshaped = X_tensor.reshape(N_stocks, T, F)
                
                # 수익률 데이터
                returns_tensor = torch.FloatTensor(returns_month).to(self.device)
                
                # 상관관계 계산
                returns_np = returns_tensor.cpu().numpy()
                if NUMBA_AVAILABLE:
                    corr_matrix = self._compute_correlation_fast_numba(returns_np.astype(np.float32))
                else:
                    corr_matrix = np.abs(np.corrcoef(returns_np))
                
                # 인접 행렬 생성
                adj = (corr_matrix > self.corr_threshold).astype(np.float32)
                adj_tensor = torch.FloatTensor(adj).to(self.device)
                
                # edge_index 생성
                rows, cols = torch.where(adj_tensor > 0)
                mask = rows != cols
                rows = rows[mask]
                cols = cols[mask]
                if len(rows) == 0:
                    continue  # 엣지가 없으면 건너뜀
                edge_index = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])])
                
                # Node features: [N, F]
                node_features = torch.mean(X_reshaped, dim=1)
                
                # GAT 처리
                batch_indices = torch.zeros(N_stocks, dtype=torch.long, device=self.device)
                h = self.model.gat1(node_features, edge_index)
                h = torch.nn.functional.relu(h)
                h = torch.nn.functional.dropout(h, p=0.2, training=True)
                h = self.model.gat2(h, edge_index)
                
                # Pooling: [N, F] → [1, F]
                h_pooled = global_mean_pool(h, batch_indices)
                
                # GRU
                h = h_pooled.unsqueeze(1)
                h, _ = self.model.gru(h)
                
                # Output: [1, 3]
                out = self.model.fc(h[:, -1, :])
                outputs_list.append(out)
                
                # y_month가 배열이면 첫 원소만 사용 (종목별 예측이 아닌 집계 예측)
                if isinstance(y_month, np.ndarray):
                    y_value = y_month[0] if len(y_month) > 0 else 0
                else:
                    y_value = y_month[0] if len(y_month) > 0 else 0
                y_list.append(torch.tensor([y_value], dtype=torch.long, device=self.device))
            
            if len(outputs_list) > 0:
                outputs = torch.cat(outputs_list, dim=0)
                y_batch = torch.cat(y_list, dim=0)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
            else:
                print(f"  [WARNING] Epoch {epoch+1}: 유효한 샘플 없음")
                break
        
        return self
    
    def fit_by_month_independent(self, X_all: List[np.ndarray], y_all: List[np.ndarray], 
                                   returns_all: List[np.ndarray], month_ends: List[pd.Timestamp]):
        """월말별로 독립적인 샘플로 학습 (각 월말은 하나의 전체 샘플)
        
        Args:
            X_all: 월말별 종목 데이터 리스트 [[종목수1, T*F], [종목수2, T*F], ...]
            y_all: 월말별 레이블 리스트
            returns_all: 월말별 수익률 리스트
            month_ends: 월말 날짜 리스트
        """
        print(f"  [INFO] 월말별 독립 샘플 처리 ({len(month_ends)}개 월말)")
        
        # 각 월말을 독립적인 샘플로 처리
        X_samples = []  # 각 월말의 종목들을 하나의 샘플로
        y_samples = []  # 각 월말의 레이블
        
        for month_end, X_month, y_month, returns_month in zip(month_ends, X_all, y_all, returns_all):
            N_stocks = X_month.shape[0]
            T, F = self.T, X_month.shape[1] // self.T
            
            if N_stocks < 2:
                continue
            
            # 전체 월말 데이터를 하나의 샘플로 추가
            X_samples.append(X_month)  # [종목수, T*F]
            y_samples.append(y_month)  # [종목수]
        
        if not X_samples:
            raise ValueError("유효한 월말 샘플이 없습니다.")
        
        # 모든 월말 데이터를 합치기
        X_combined = np.vstack(X_samples)  # [총샘플수, T*F]
        y_combined = np.concatenate(y_samples)  # [총샘플수]
        
        # 수익률 데이터도 합치기
        returns_combined = np.vstack(returns_all)  # [총샘플수, T]
        
        # 일반 fit 호출
        self.fit(X_combined, y_combined, returns_data=returns_combined)
        
        return self
    
    def fit_by_single_month(self, X_month, y_month, returns_month, month_end):
        """단일 월말 기준일의 데이터로 학습
        
        Args:
            X_month: [N종목, T*F] 월말의 종목 데이터
            y_month: [N종목] 월말의 레이블
            returns_month: [N종목, T] 월말의 수익률
            month_end: 월말 날짜
        """
        print(f"  [INFO] 월말 기준일 {month_end.date()} 학습: N={len(y_month)} 종목")
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X_month)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_month).to(self.device)
        
        N_stocks = X_tensor.shape[0]
        T, F = self.T, int(X_tensor.shape[1] // self.T)
        
        if F <= 0:
            raise ValueError(f"Invalid F: {F} (shape={X_tensor.shape}, T={self.T})")
        
        # Reshape: [N, T*F] -> [N, T, F]
        X_reshaped = X_tensor.reshape(N_stocks, T, F)
        
        # 상관관계 행렬 계산 (N×N)
        returns_tensor = torch.FloatTensor(returns_month).to(self.device)
        returns_np = returns_tensor.cpu().numpy()
        
        # 데이터 차원 확인 및 검증
        print(f"  [DEBUG] returns_np shape: {returns_np.shape}")
        if returns_np.ndim == 1:
            raise ValueError(f"returns_np is 1D array (shape={returns_np.shape}), expected 2D [N, T]")
        
        if NUMBA_AVAILABLE:
            corr_matrix = self._compute_correlation_fast_numba(returns_np.astype(np.float32))
        else:
            corr_matrix = np.abs(np.corrcoef(returns_np))
        
        # 엣지 생성
        adj = (corr_matrix > self.corr_threshold).astype(np.float32)
        adj_tensor = torch.FloatTensor(adj).to(self.device)
        
        rows, cols = torch.where(adj_tensor > 0)
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        
        if len(rows) == 0:
            print(f"  [WARNING] 엣지가 없음 (상관관계 모두 낮음)")
            return self
        
        edge_index = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])])
        
        # Model initialization
        print(f"  [DEBUG] Initializing model with F={F}, hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, num_heads={self.num_heads}")
        print(f"  [DEBUG] F type: {type(F)}, value: {F}")
        
        # Ensure all parameters are integers
        num_features_int = int(F)
        hidden_dim_int = int(self.hidden_dim)
        num_layers_int = int(self.num_layers)
        num_heads_int = int(self.num_heads)
        
        print(f"  [DEBUG] After conversion - num_features={num_features_int}, hidden_dim={hidden_dim_int}, num_layers={num_layers_int}, num_heads={num_heads_int}")
        
        self.model = TemporalGATGRU(num_features_int, hidden_dim_int, num_layers_int, num_heads_int)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Node features: [N, F]
            node_features = torch.mean(X_reshaped, dim=1)
            
            # GAT 처리
            batch_indices = torch.zeros(N_stocks, dtype=torch.long, device=self.device)
            h = self.model.gat1(node_features, edge_index)
            h = torch.nn.functional.relu(h)
            h = torch.nn.functional.dropout(h, p=0.2, training=True)
            h = self.model.gat2(h, edge_index)
            
            # Pooling: [N, hidden_dim] -> [N, hidden_dim] (각 종목별 예측)
            h_pooled = h  # 각 종목별로 예측
            
            # GRU
            h = h_pooled.unsqueeze(1)  # [N, 1, hidden_dim]
            h, _ = self.model.gru(h)
            
            # Output: [N, 3]
            out = self.model.fc(h[:, -1, :])
            
            # Loss 계산 (각 종목별)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def predict_proba(self, X, returns_data=None):
        """Predict class probabilities (optimized)
        
        Args:
            X: [batch_size, T*F] input features
            returns_data: [batch_size, T] returns data for correlation (optional)
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Precompute graphs for all samples
        X_batch_list = []
        edge_index_batch_list = []
        
        for i in range(X_tensor.shape[0]):
            X_sample = X_tensor[i:i+1]
            # 수익률 데이터가 있으면 함께 전달
            if returns_data is not None:
                returns_sample = returns_data[i:i+1]  # [1, T]
                X_reshaped_sample, edge_index_sample = self._build_graph(X_sample, returns_sample)
            else:
                X_reshaped_sample, edge_index_sample = self._build_graph(X_sample)
            node_features = torch.mean(X_reshaped_sample, dim=1)
            
            X_batch_list.append(node_features)
            edge_index_batch_list.append(edge_index_sample)
        
        with torch.no_grad():
            outputs_list = []
            
            # Process all samples with pre-computed graphs
            for i in range(len(X_batch_list)):
                node_features = X_batch_list[i]
                edge_index = edge_index_batch_list[i]
                
                # Create batch indices
                batch_indices = torch.zeros(node_features.shape[0], dtype=torch.long, device=self.device)
                
                # GAT forward
                h = self.model.gat1(node_features, edge_index)
                h = torch.nn.functional.relu(h)
                h = self.model.gat2(h, edge_index)
                
                # Pooling
                h = global_mean_pool(h, batch_indices)
                
                # GRU
                h = h.unsqueeze(1)
                h, _ = self.model.gru(h)
                
                # Output
                out = self.model.fc(h[:, -1, :])
                outputs_list.append(out)
            
            outputs = torch.cat(outputs_list, dim=0)
            probs = F.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def make_model(T=20) -> TemporalGATWrapper:
    """Return a Temporal-GAT + GRU model wrapper"""
    return TemporalGATWrapper(T=T, hidden_dim=64, num_layers=2, num_heads=4)


# -------------------------- Train ---------------------------- #

def train_month_end_models(df: pd.DataFrame, asof: pd.Timestamp, save_dir: str = None, model_idx: int = None) -> Dict[int, str]:
    """Train one model per window T using daily samples up to `asof`.
    Each sample uses past T days as input and predicts next 20 days return.
    Returns mapping T -> model_path saved.
    """
    ensure_dirs()
    assert_trading_cal_ready(df)
    
    # save_dir가 제공되면 해당 디렉토리 생성
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    all_dates = sorted(df["date"].unique())
    if asof not in all_dates:
        raise ValueError(f"asof {asof.date()} not found in data dates.")

    # asof 기준일 하나만 사용 (이전 월말들 수집하지 않음)
    min_data_date = pd.Timestamp("2021-04-30")
    label_horizon = pd.tseries.offsets.BDay(20)
    
    print(f"  [DEBUG] asof: {asof}")

    symbols = sorted(df["symbol"].unique().tolist())
    saved = {}
    
    # 종목별 데이터 캐싱 (성능 향상)
    symbol_groups = group_data_by_symbol(df)
    print(f"  [INFO] 종목별 데이터 그룹 생성: {len(symbol_groups)}개 종목")

    for T in WINDOWS:
        # 80 business days policy check
        min_span = pd.bdate_range(asof - pd.tseries.offsets.BDay(79), asof)
        if len(min_span) < 80:
            raise ValueError("Need at least 80 business days before asof.")

        # asof 기준일 하나만 사용
        print(f"  [INFO] asof 기준 T={T} 학습: {asof.date()}")
        
        # asof 기준으로 윈도우 데이터 추출
        start = asof - pd.tseries.offsets.BDay(T - 1)
        print(f"    윈도우: {start.date()} ~ {asof.date()} ({T} 영업일)")
        X_asof, mask_asof, kept_syms, returns_asof = build_window_matrix(
            df, symbols, asof, T, FEATURES, symbol_groups
        )
        
        # 데이터 검증
        print(f"    [DEBUG] X_asof shape: {X_asof.shape}, returns_asof shape: {returns_asof.shape}")
        
        if not mask_asof.any():
            print(f"    [WARNING] asof {asof.date()} 유효한 샘플 없음 (유효: {mask_asof.sum()}/{len(mask_asof)})")
            # saved[T]를 비워두지 않고 건너뛰기
            saved[T] = None  # None으로 표시하여 예측 단계에서 건너뜀
            continue
        
        # 레이블 생성 (asof 기준)
        y_asof_full = build_label_vector(df, symbols, asof)
        y_asof = y_asof_full[mask_asof]
        X_asof_masked = X_asof[mask_asof]
        returns_asof_masked = returns_asof[mask_asof]
        
        # 데이터 검증
        print(f"    [DEBUG] After masking - X_asof_masked: {X_asof_masked.shape}, returns_asof_masked: {returns_asof_masked.shape}, y_asof: {y_asof.shape}")
        
        # asof 기준일 하나의 종목들만 학습 (N×F 노드, N×N 엣지)
        model = make_model(T=T)
        model.fit_by_single_month(X_asof_masked, y_asof, returns_asof_masked, asof)
        
        # 모델 저장
        if save_dir is not None:
            if model_idx is not None:
                model_path = os.path.join(save_dir, f"model_T{T}_{model_idx}.joblib")
            else:
                model_path = os.path.join(save_dir, f"model_T{T}.joblib")
        else:
            model_path = os.path.join(MODEL_DIR, f"model_T{T}_{asof.strftime('%Y%m%d')}.joblib")
        
        joblib.dump({"model": model, "features": FEATURES, "T": T, "asof": asof}, model_path)
        saved[T] = model_path
        print(f"Saved: {model_path}  (N={len(y_asof)} 종목, asof={asof.date()})")

    return saved


# ------------------------- Predict --------------------------- #

def predict_next_month(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Load models for the given asof, predict per T and ensemble, then export Excel."""
    ensure_dirs()
    symbols = sorted(df["symbol"].unique().tolist())
    assert_trading_cal_ready(df)

    results: Dict[int, Dict[str, np.ndarray]] = {}
    
    # 종목별 데이터 캐싱
    symbol_groups = group_data_by_symbol(df)

    for T in WINDOWS:
        model_path = os.path.join(MODEL_DIR, f"model_T{T}_{asof.strftime('%Y%m%d')}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Train first.")
        pack = joblib.load(model_path)
        # Handle both Pipeline and TemporalGATWrapper
        model = pack["model"]

        X, mask, kept, returns = build_window_matrix(df, symbols, asof, T, FEATURES, symbol_groups)
        # Predict only for valid rows
        proba = np.full((len(symbols), 3), np.nan, dtype=float)
        if mask.any():
            # Check if model supports returns_data parameter
            if hasattr(model, 'predict_proba') and 'returns_data' in model.predict_proba.__code__.co_varnames:
                p = model.predict_proba(X[mask], returns_data=returns[mask])
            else:
                p = model.predict_proba(X[mask])
            # Handle different output formats
            # TemporalGATWrapper or sklearn Pipeline both work
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

def train_and_predict_ensemble(df: pd.DataFrame, n_models: int = 5) -> pd.DataFrame:
    """월말 영업일 기준 반복 학습 및 예측 (5개 모델 앙상블)
    
    Args:
        df: 전처리된 데이터프레임
        n_models: 앙상블할 모델 개수 (기본 5개)
    
    Returns:
        예측 결과 데이터프레임 [월말 영업일 X 종목코드]
    """
    # 실행 시간 기반 타이틀 생성
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join("results", run_timestamp)
    
    print("\n" + "="*80)
    print("월말 영업일 기준 반복 학습 및 예측 시작")
    print(f"결과 저장 디렉토리: {run_dir}")
    print("="*80)
    
    # 모든 월말 영업일 추출
    assert_trading_cal_ready(df)
    month_ends = get_month_ends(TRADING_CAL)
    
    if len(month_ends) < 2:
        raise ValueError("최소 2개 이상의 월말 영업일이 필요합니다.")
    
    # 학습 범위: 4번째 월말부터 마지막에서 3번째까지
    if len(month_ends) >= 7:  # 최소 7개 월말 필요 (4번째부터 시작)
        train_month_ends = month_ends[3:-3]  # 4번째(인덱스 3)부터 마지막-3번째까지
        predict_month_ends = month_ends[4:-2]  # 5번째부터 마지막-2번째까지 (예측)
    else:
        train_month_ends = []
        predict_month_ends = []
    
    if len(train_month_ends) == 0:
        raise ValueError(f"학습 가능한 월말이 없습니다. 최소 7개 월말이 필요하며, 현재 {len(month_ends)}개입니다.")
    
    print(f"총 {len(train_month_ends)}개 월말 영업일에서 학습/예측 진행")
    print(f"학습 기간: {train_month_ends[0]} ~ {train_month_ends[-1]}")
    print(f"예측 기간: {predict_month_ends[0]} ~ {predict_month_ends[-1]}")
    print()
    
    # 최종 예측 결과 저장
    all_predictions = []
    
    # 기준일별 모델 저장 디렉토리 생성
    date_dirs = {}  # {train_asof: base_dir}
    
    # 각 월말 영업일마다 반복
    for idx, (train_asof, predict_asof) in enumerate(zip(train_month_ends, predict_month_ends)):
        # 기준일별 폴더 생성
        date_str = train_asof.strftime('%Y%m%d')
        date_dir = os.path.join(run_dir, f"models_{date_str}")
        date_dirs[train_asof] = date_dir
        os.makedirs(date_dir, exist_ok=True)
        print(f"\n[{idx+1}/{len(train_month_ends)}] 학습: {train_asof.strftime('%Y-%m-%d')}, 예측: {predict_asof.strftime('%Y-%m-%d')}")
        print("-" * 80)
        
        # 5개 모델 생성 및 학습
        ensemble_predictions = []
        
        for seed in range(n_models):
            # 랜덤 시드 설정
            random.seed(42 + seed)
            np.random.seed(42 + seed)
            torch.manual_seed(42 + seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42 + seed)
            
            print(f"  [모델 {seed+1}/{n_models}] 학습 중...", end=" ", flush=True)
            
            try:
                # 현재 월말 영업일 기준 학습 (기준일별 폴더에 모델 번호로 저장)
                saved = train_month_end_models(df, train_asof, save_dir=date_dir, model_idx=seed)
                
                # 종목 리스트
                symbols = sorted(df["symbol"].unique().tolist())
                
                # 직후 월말 영업일 기준 예측
                X, mask, kept, returns = build_window_matrix(df, symbols, predict_asof, WINDOWS[0], FEATURES)
                
                if not mask.any():
                    print("(유효 데이터 없음)")
                    ensemble_predictions.append(None)
                    continue

                # 모든 윈도우에 대해 예측 및 앙상블
                all_probas_full = []  # 각 원소: (len(symbols), 3) with NaN for missing rows

                for win in WINDOWS:
                    if saved[win] is None:
                        # 이 윈도우의 모델이 없으면 통째로 NaN 채움
                        proba_full = np.full((len(symbols), 3), np.nan, dtype=float)
                        all_probas_full.append(proba_full)
                        continue

                    model_path = saved[win]
                    pack = joblib.load(model_path)
                    model = pack["model"]

                    X_w, mask_w, kept_w, returns_w = build_window_matrix(df, symbols, predict_asof, win, FEATURES)

                    # 심볼 전체 길이에 맞춰서 채우기
                    proba_full = np.full((len(symbols), 3), np.nan, dtype=float)

                    if mask_w.any():
                        if hasattr(model,
                                   "predict_proba") and "returns_data" in model.predict_proba.__code__.co_varnames:
                            proba_w = model.predict_proba(X_w[mask_w], returns_data=returns_w[mask_w])
                        else:
                            proba_w = model.predict_proba(X_w[mask_w])

                        # mask_w가 True인 위치(kept_w 인덱스)에만 채워 넣기
                        proba_full[mask_w] = proba_w
                    # mask_w가 전부 False면 proba_full은 그대로 NaN

                    all_probas_full.append(proba_full)

                # NaN 무시 평균 (심볼별/클래스별)
                avg_proba = np.nanmean(np.stack(all_probas_full, axis=0), axis=0)  # shape: (len(symbols), 3)

                # 예측 클래스
                pred_full = np.argmax(avg_proba, axis=1)

                # 원래 mask(최초 WINDOWS[0] 기준)로 유효 심볼만 취해도 되고, pred_full 전체를 써도 됨
                # 여기서는 최초 mask 기준으로만 결과를 사용
                pred_masked = np.full(len(symbols), np.nan)
                pred_masked[mask] = pred_full[mask]

                ensemble_predictions.append(pred_masked)
                
                print("완료")
                
            except Exception as e:
                print(f"에러: {e}")
                ensemble_predictions.append(None)
        
        # 5개 모델 예측값 평균 계산
        valid_predictions = [p for p in ensemble_predictions if p is not None]
        
        if valid_predictions:
            # 각 모델의 예측값을 종합 (평균값 사용)
            symbols = sorted(df["symbol"].unique().tolist())
            ensemble_array = np.array(valid_predictions).T  # [종목, 모델]
            final_pred = np.round(np.nanmean(ensemble_array, axis=1)).astype(int)
            
            # 결과 저장
            symbols = sorted(df["symbol"].unique().tolist())
            result_df = pd.DataFrame({
                "date": [predict_asof.strftime('%Y-%m-%d')] * len(symbols),
                "symbol": symbols,
                "prediction": final_pred
            })
            all_predictions.append(result_df)
            
            print(f"  [완료] 평균 예측값 계산 ({len(valid_predictions)}개 모델 유효)")
        else:
            print(f"  [경고] 유효한 예측 없음")
    
    # 모든 결과 통합
    if not all_predictions:
        raise ValueError("예측 결과가 없습니다.")
    
    final_results = pd.concat(all_predictions, ignore_index=True)
    
    # 피벗: [월말 영업일 X 종목코드] 행렬
    pivot_table = final_results.pivot(index='date', columns='symbol', values='prediction')
    
    # 엑셀 저장 (실행시간 타이틀 폴더에 저장)
    output_path = os.path.join(run_dir, "ensemble_predictions.xlsx")
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pivot_table.to_excel(writer, sheet_name="predictions_matrix")
        final_results.to_excel(writer, sheet_name="predictions_long")
    
    print(f"\n{'='*80}")
    print(f"최종 결과 저장: {output_path}")
    print(f"총 {len(final_results)}개 예측 (월말: {final_results['date'].nunique()}개)")
    print(f"{'='*80}\n")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Temporal GAT 월말 영업일 반복 학습/예측",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        사용 예시:
          
          # Ensemble 모드 (추천): 월말 영업일별 학습/예측 자동 실행
          python temporal_gat_monthly_ensemble_pipeline_patched.py --mode ensemble
          
          # Ensemble 모드 (모델 개수 지정)
          python temporal_gat_monthly_ensemble_pipeline_patched.py --mode ensemble --n-models 5
          
          # Ensemble 모드 (커스텀 데이터 파일)
          python temporal_gat_monthly_ensemble_pipeline_patched.py --mode ensemble --data preprocessed_data.xlsx
          
          # 학습 모드: 특정 날짜 기준 학습
          python temporal_gat_monthly_ensemble_pipeline_patched.py --mode train --asof 2021-04-30
          
          # 예측 모드: 특정 날짜 기준 예측
          python temporal_gat_monthly_ensemble_pipeline_patched.py --mode predict --asof 2021-04-30
        
        설명:
          - ensemble: '2021-04-30'부터 자동으로 월말 영업일 기준 반복 학습/예측 수행
          - train: 지정된 날짜 기준으로 모델 학습
          - predict: 지정된 날짜 기준으로 예측 수행
        """
    )
    parser.add_argument("--mode", choices=["train", "predict", "ensemble", "debug"], default="ensemble",
                        help="실행 모드: train(학습), predict(예측), ensemble(반복 학습/예측, 기본값), debug(입출력 데이터 추출)")
    parser.add_argument("--asof", type=str, default=None,
                        help="Month-end date (YYYY-MM-DD) - train/predict 모드용")
    parser.add_argument("--data", type=str, default=None, help="데이터 파일 경로 (기본: DATA_PATH)")
    parser.add_argument("--n-models", type=int, default=5, help="앙상블 모델 개수 (기본: 5)")
    args = parser.parse_args()

    # 데이터 파일 경로 결정
    data_path = args.data or DATA_PATH
    if not args.data and os.path.exists(PREPROCESSED_DATA_PATH):
        data_path = PREPROCESSED_DATA_PATH

    df = load_and_engineer(data_path)

    if args.mode == "debug":
        if not args.asof:
            raise ValueError("--asof 필요 (YYYY-MM-DD)")
        
        try:
            asof = month_end(parse_date(args.asof))
            print(f"[DEBUG] asof 변환: {args.asof} -> {asof}", flush=True)
        except Exception as e:
            print(f"[ERROR] 날짜 변환 실패: {e}", flush=True)
            raise
        
        # 2021-05-31 기준 입력값과 출력값을 엑셀로 추출
        print(f"\n{'='*80}", flush=True)
        print(f"디버그 모드: {asof.date()} 기준 입출력 데이터 추출", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        symbols = sorted(df["symbol"].unique().tolist())
        print(f"기준일: {asof.date()}", flush=True)
        print(f"전체 종목 수: {len(symbols)}", flush=True)
        print(f"데이터 날짜 범위: {df['date'].min()} ~ {df['date'].max()}", flush=True)
        
        # 종목별 데이터 그룹
        symbol_groups = group_data_by_symbol(df)
        print(f"종목별 데이터 그룹 생성: {len(symbol_groups)}개 종목", flush=True)
        # 첫 번째 종목 데이터 확인
        if symbol_groups:
            first_symbol = list(symbol_groups.keys())[0]
            first_df = symbol_groups[first_symbol]
            print(f"첫 번째 종목({first_symbol}) 데이터 범위: {first_df.index.min()} ~ {first_df.index.max()}, 행 수: {len(first_df)}", flush=True)
        
        # 입력값 추출 (T=20, 40, 60)
        results = {}
        
        for T in WINDOWS:
            print(f"\n=== T={T} 영업일 ===")
            
            # 윈도우 기간 확인
            start = asof - pd.tseries.offsets.BDay(T - 1)
            wdates = pd.bdate_range(start, asof)
            print(f"  필요한 기간: {start.date()} ~ {asof.date()} ({len(wdates)} 영업일)")
            
            X, mask, kept, returns = build_window_matrix(df, symbols, asof, T, FEATURES, symbol_groups)
            
            print(f"  입력값 shape: {X.shape}")
            print(f"  유효 샘플 수: {mask.sum()}/{len(mask)}")
            print(f"  유효 종목: {len(kept)}")
            
            if mask.sum() > 0:
                # 입력값 엑셀로 저장
                X_df = pd.DataFrame(X[mask], columns=[f"F{i}" for i in range(X.shape[1])])
                X_df.insert(0, "symbol", np.array(kept))
                X_df.insert(1, "asof", asof.date())
                X_df.insert(2, "window_days", T)
                
                # returns 데이터도 추가
                returns_df = pd.DataFrame(returns[mask], columns=[f"ret_day_{i}" for i in range(returns.shape[1])])
                X_full = pd.concat([X_df, returns_df], axis=1)
                
                results[f"X_T{T}"] = X_full
        
        # 출력값 (레이블) 추출
        y_full = build_label_vector(df, symbols, asof)
        label_map = {0: "하락", 1: "보합", 2: "상승"}
        y_df = pd.DataFrame({
            "symbol": symbols,
            "asof": asof.date(),
            "label": y_full,
            "label_name": [label_map.get(l, "알수없음") for l in y_full]
        })
        
        results["y_labels"] = y_df
        
        # 엑셀로 저장
        output_path = f"debug_{asof.strftime('%Y%m%d')}_data.xlsx"
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            results["y_labels"].to_excel(writer, sheet_name="레이블", index=False)
            for T in WINDOWS:
                if f"X_T{T}" in results:
                    results[f"X_T{T}"].to_excel(writer, sheet_name=f"입력_T{T}", index=False)
        
        print(f"\n{'='*80}")
        print(f"저장 완료: {output_path}")
        print(f"  - 레이블: {len(y_df)}개 종목")
        for T in WINDOWS:
            if f"X_T{T}" in results:
                print(f"  - T={T} 입력: {len(results[f'X_T{T}'])}개 유효 종목")
        print(f"{'='*80}\n")

    elif args.mode == "ensemble":
        # 반복 학습/예측 모드
        train_and_predict_ensemble(df, n_models=args.n_models)

    elif args.mode == "train":
        if not args.asof:
            raise ValueError("--asof 필요 (YYYY-MM-DD)")
        asof = month_end(parse_date(args.asof))
        # 80 영업일 가드
        if len(pd.bdate_range(asof - pd.tseries.offsets.BDay(79), asof)) < 80:
            raise SystemExit("ERROR: Need at least 80 business days of data prior to asof.")
        train_month_end_models(df, asof)

    elif args.mode == "predict":
        if not args.asof:
            raise ValueError("--asof 필요 (YYYY-MM-DD)")
        asof = month_end(parse_date(args.asof))
        # 80 영업일 가드
        if len(pd.bdate_range(asof - pd.tseries.offsets.BDay(79), asof)) < 80:
            raise SystemExit("ERROR: Need at least 80 business days of data prior to asof.")
        predict_next_month(df, asof)


    else:
        raise ValueError("invalid --mode")


if __name__ == "__main__":
    main()
