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


def get_month_ends(dts: List[pd.Timestamp]) -> List[pd.Timestamp]:
    """월말 영업일 추출 - 매월 마지막 영업일 기준 (2021-04-30 이후만)"""
    import pandas as pd
    
    # 2021-04-30 이후 데이터만 사용
    min_date = pd.Timestamp("2021-04-30")
    filtered_dts = [d for d in dts if d >= min_date]
    
    if not filtered_dts:
        return []
    
    # 각 월별로 마지막 날짜 찾기
    df = pd.DataFrame(filtered_dts, columns=['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    month_ends = []
    for period in df['year_month'].unique():
        month_data = df[df['year_month'] == period]['date']
        # 영업일만 필터링 (주말 제외)
        business_days = month_data[month_data.dt.weekday < 5]
        if len(business_days) > 0:
            month_ends.append(business_days.max())
    
    return sorted(month_ends)


def next_month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthEnd(1)).normalize()


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
        df_price = pd.read_excel(data_path, sheet_name="price")
        df_volume = pd.read_excel(data_path, sheet_name="volume")

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


def build_window_matrix(
    df: pd.DataFrame,
    symbols: List[str],
    asof: pd.Timestamp,
    T: int,
    features: List[str] = FEATURES,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build [N, T*F] matrix for a single asof date.
    Returns (X, mask, symbols_kept) where mask marks rows that are fully valid.
    
    Also extracts returns separately for correlation computation.
    """
    start = asof - pd.tseries.offsets.BDay(T - 1)
    # filter window rows - all features
    wdf = df[(df["date"] >= start) & (df["date"] <= asof)][["date", "symbol"] + features]
    wdates = pd.bdate_range(start, asof)

    X_list = []
    returns_list = []  # 수익률 데이터 분리
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
            returns_list.append(np.zeros((T,), dtype=np.float32))
            continue
        kept.append(sym)
        mask.append(True)
        X_list.append(sdf.values.astype(np.float32))
        
        # log_return만 추출 (첫 번째 feature)
        if "log_return" in features:
            returns_list.append(sdf["log_return"].values.astype(np.float32))
        else:
            # features 첫 번째가 log_return인지 확인
            returns_list.append(sdf.iloc[:, 0].values.astype(np.float32))

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
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
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
            X_flat: [batch_size, T*F] 전체 특성
            returns_flat: [batch_size, T] 수익률 데이터 (선택적)
        
        Returns:
            X: [batch_size, T, F] reshaped features
            edge_index: [2, num_edges]
        """
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
        
    def fit(self, X, y, returns_data=None):
        """Train the model
        
        Args:
            X: [batch_size, T*F] input features
            y: [batch_size] labels
            returns_data: [batch_size, T] returns data for correlation (optional)
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Model initialization
        num_features = X_tensor.shape[1] // self.T
        self.model = TemporalGATGRU(num_features, self.hidden_dim, self.num_layers, self.num_heads)
        self.model.to(self.device)
        
        # Batch preprocessing for efficiency
        print("  [INFO] Preprocessing graphs with returns correlation...")
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
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop with batch processing
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Process all samples with pre-computed graphs
            outputs_list = []
            
            for i in range(len(X_batch_list)):
                node_features = X_batch_list[i]
                edge_index = edge_index_batch_list[i]
                
                # Create batch indices
                batch_indices = torch.zeros(node_features.shape[0], dtype=torch.long, device=self.device)
                
                # GAT forward (batch efficiently)
                h = self.model.gat1(node_features, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=0.2, training=True)
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
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities (optimized)"""
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
                h = F.relu(h)
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
    all_dates = sorted(df["date"].unique())
    if asof not in all_dates:
        raise ValueError(f"asof {asof.date()} not found in data dates.")

    # restrict to dates up to asof - we need 20 BDays after for label
    # So available samples: from 2021-04-30 to (asof - 20 BDays)
    min_data_date = pd.Timestamp("2021-04-30")
    max_training_date = asof - pd.tseries.offsets.BDay(20)
    trainable_dates = [d for d in all_dates if min_data_date <= d <= max_training_date]
    
    # 월말 기준으로 샘플링 (하루마다 학습하면 너무 많음)
    month_ends = get_month_ends(trainable_dates)

    symbols = sorted(df["symbol"].unique().tolist())
    saved = {}

    for T in WINDOWS:
        # 80 business days policy check
        min_span = pd.bdate_range(asof - pd.tseries.offsets.BDay(79), asof)
        if len(min_span) < 80:
            raise ValueError("Need at least 80 business days before asof.")

        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        returns_all: List[np.ndarray] = []  # 수익률 데이터 수집

        for me in month_ends:
            # require we can also label (need 20 BDays after)
            future_date = me + pd.tseries.offsets.BDay(20)
            if future_date not in all_dates:
                continue

            X_me, mask_me, kept_syms, returns_me = build_window_matrix(df, symbols, me, T)
            if not mask_me.any():
                continue
            y_me_full = build_label_vector(df, symbols, me)
            y_me = y_me_full[mask_me]
            X_me = X_me[mask_me]
            returns_me = returns_me[mask_me]  # 유효한 종목만

            X_all.append(X_me)
            y_all.append(y_me)
            returns_all.append(returns_me)  # 수익률도 함께 저장

        if not X_all:
            raise ValueError(f"No training samples for T={T} up to {asof.date()}.")

        Xtr = np.vstack(X_all)
        ytr = np.concatenate(y_all)
        returns_tr = np.vstack(returns_all)  # 수익률 데이터

        model = make_model(T=T)
        model.fit(Xtr, ytr, returns_data=returns_tr)
        
        # 모델 저장 경로 결정
        if save_dir is not None:
            # 새로운 구조: save_dir/model_T{T}_{model_idx}.joblib
            if model_idx is not None:
                model_path = os.path.join(save_dir, f"model_T{T}_{model_idx}.joblib")
            else:
                model_path = os.path.join(save_dir, f"model_T{T}.joblib")
        else:
            # 기존 구조: MODEL_DIR/model_T{T}_{asof}.joblib
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
        # Handle both Pipeline and TemporalGATWrapper
        model = pack["model"]

        X, mask, kept, returns = build_window_matrix(df, symbols, asof, T)
        # Predict only for valid rows
        proba = np.full((len(symbols), 3), np.nan, dtype=float)
        if mask.any():
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
    all_dates = sorted(df["date"].unique())
    month_ends = get_month_ends(all_dates)
    
    if len(month_ends) < 2:
        raise ValueError("최소 2개 이상의 월말 영업일이 필요합니다.")
    
    # 마지막 월말 영업일 직전까지 학습/예측 반복
    train_month_ends = month_ends[:-1]  # 마지막 직전까지
    predict_month_ends = month_ends[1:]  # 두 번째부터 예측
    
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
                
                # 직후 월말 영업일 기준 예측
                X, mask, kept, returns = build_window_matrix(df, sorted(df["symbol"].unique().tolist()), predict_asof, WINDOWS[0])
                
                if not mask.any():
                    print("(유효 데이터 없음)")
                    ensemble_predictions.append(None)
                    continue
                
                # 각 T에 대해 예측
                model_path = saved[WINDOWS[0]]
                pack = joblib.load(model_path)
                model = pack["model"]
                
                # 예측 실행
                proba = model.predict_proba(X[mask], returns_data=returns[mask])
                pred = np.argmax(proba, axis=1)
                
                # 전체 종목에 대해 NaN으로 초기화 후 예측값 채우기
                symbols = sorted(df["symbol"].unique().tolist())
                pred_full = np.full(len(symbols), np.nan)
                pred_full[mask] = pred
                ensemble_predictions.append(pred_full)
                
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
    parser = argparse.ArgumentParser(description="Temporal GAT 월말 영업일 반복 학습/예측")
    parser.add_argument("--mode", choices=["train", "predict", "ensemble"], required=True,
                        help="실행 모드: train(학습), predict(예측), ensemble(반복 학습/예측)")
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

    if args.mode == "ensemble":
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
