"""
Monthly 20/40/60 Ensemble Pipeline (GPU-enabled, Temporal-GAT + GRU)
---------------------------------------------------------------------
- Windows: T in {20, 40, 60}; labels: 2 (>=+10%), 1 (+5%~<+10%), 0 (else)
- Train on each month-end (asof). Save model per T with timestamp.
- Later, load saved models for that same asof date and produce next-month-end
  predictions, exporting a per-symbol matrix to Excel.

This version targets CUDA when available and falls back to CPU automatically.
Key GPU points:
- Unified DEVICE = torch.device('cuda' if available else 'cpu')
- All tensors moved to DEVICE via .to(DEVICE)
- Optional AMP (torch.cuda.amp) enabled for forward/backward
- DataLoader uses pin_memory=True and non_blocking transfers when on CUDA
- Safe load with map_location, and model.to(DEVICE) after loading

Inputs (Excel):
- Either a preprocessed long-format Excel (sheet "data": date, symbol, close, volume[, engineered features])
- Or a wide matrix Excel with sheets "price" and "volume" (first column = date)

Outputs:
- Models: models/{YYYYMMDD}/T{T}_seed{seed}.joblib
- Predictions: output/pred_matrix_{YYYYMMDD}.xlsx (symbols × predictions)

CLI:
  Train + Predict rolling month-ends
    python temporal_gat_monthly_ensemble_pipeline_GPU.py --data preprocessed_data.xlsx \
      --start 2021-07-30 --end 2025-07-31 --n-models 5 --epochs 50

  Single run (train only or predict only) is supported via --mode
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.utils.class_weight import compute_class_weight
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# ------------------------ CUDA / Determinism ------------------------ #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # perf (ok for variable shapes)

# ------------------------ Global Config ----------------------------- #
DATA_PATH = "preprocessed_data.xlsx"
MODEL_DIR = "models"
PRED_DIR = "output"
WINDOWS = [20, 40, 60]
FEATURES = ["log_return", "vol_z", "volatility20", "mom20"]

# Trading calendar derived from Excel (actual KRX trading days)
TRADING_CAL: Optional[pd.DatetimeIndex] = None

# ------------------------ Calendar utilities ----------------------- #
def _ensure_datetime_series(s: pd.Series) -> pd.Series:
    """Convert Excel 'date' safely to naive normalized Timestamps.
    Handles strings, timestamps, and Excel serial numbers.
    """
    s1 = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if s1.isna().mean() > 0.8 and pd.api.types.is_numeric_dtype(s):
        s1 = pd.to_datetime(s.astype("float64"), unit="d", origin="1899-12-30", errors="coerce").dt.normalize()
    return s1

def set_trading_calendar_from_df(df: pd.DataFrame) -> None:
    global TRADING_CAL
    dates = _ensure_datetime_series(df["date"]).dropna()
    TRADING_CAL = pd.DatetimeIndex(dates.unique()).sort_values()

def snap_to_calendar(asof: pd.Timestamp, cal: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    if cal is None or len(cal) == 0:
        return None
    a = pd.Timestamp(asof).tz_localize(None).normalize()
    pos = cal.searchsorted(a, side="right") - 1
    if pos < 0:
        return None
    return cal[pos]

def last_n_trading_days(asof: pd.Timestamp, n: int, cal: pd.DatetimeIndex) -> Optional[pd.DatetimeIndex]:
    if cal is None or len(cal) == 0:
        return None
    a = snap_to_calendar(asof, cal)
    if a is None:
        return None
    pos = cal.get_indexer([a])[0]
    start = pos - (n - 1)
    if start < 0:
        return None
    return cal[start : pos + 1]

def get_month_ends(cal: pd.DatetimeIndex) -> List[pd.Timestamp]:
    if cal is None or len(cal) == 0:
        return []
    per = cal.to_period("M")
    s = pd.Series(cal, index=per)
    ends = s.groupby(level=0).max()
    ends = ends[ends >= pd.Timestamp("2021-04-30")]
    return ends.tolist()

# ------------------------ IO helpers -------------------------------- #
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

def load_and_engineer(data_path: str) -> pd.DataFrame:
    """Load Excel and return long-format dataframe with engineered features.
    Supports either:
      (1) preprocessed long-form (sheet "data"), or
      (2) wide price/volume matrices (sheets "price", "volume").
    """
    try:
        xl = pd.ExcelFile(data_path)
        if "data" in xl.sheet_names:
            print(f"전처리된 엑셀 파일 로드: {data_path}")
            df = pd.read_excel(data_path, sheet_name="data")
            df["close"] = df["close"].fillna(0)
            df["volume"] = df["volume"].fillna(0)
            req = {"date", "symbol", "close", "volume"}
            missing = req - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            df["date"] = _ensure_datetime_series(df["date"])  # normalize
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
            if all(c in df.columns for c in FEATURES):
                return df
        # else: treat as wide matrices
        print(f"원본 엑셀 파일 처리 중: {data_path}")
        df_price = pd.read_excel(data_path, sheet_name="price").fillna(0)
        df_volume = pd.read_excel(data_path, sheet_name="volume").fillna(0)
        df_price = df_price.rename(columns={df_price.columns[0]: "date"})
        df_volume = df_volume.rename(columns={df_volume.columns[0]: "date"})
        df_price["date"] = _ensure_datetime_series(df_price["date"])  # normalize
        df_volume["date"] = _ensure_datetime_series(df_volume["date"])  # normalize
        price_long = df_price.melt(id_vars=["date"], var_name="symbol", value_name="close")
        volume_long = df_volume.melt(id_vars=["date"], var_name="symbol", value_name="volume")
        df = pd.merge(price_long, volume_long, on=["date", "symbol"], how="inner")
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Failed to load data from {data_path}: {e}")

    # Feature engineering
    df["log_return"] = np.log(df.groupby("symbol")["close"].pct_change().add(1.0))
    df["log_vol"] = np.log(df["volume"].fillna(0) + 1)
    grp = df.groupby("symbol")
    df["vol_z"] = grp["log_vol"].transform(lambda s: (s - s.rolling(60).mean()) / (s.rolling(60).std() + 1e-8))
    df["volatility20"] = grp["log_return"].transform(lambda s: s.rolling(20).std())
    df["mom20"] = grp["log_return"].transform(lambda s: s.rolling(20).sum())

    # drop rows where engineered features are NaN
    df = df.dropna(subset=FEATURES)
    # use only 2021-04-30+
    df = df[df["date"] >= pd.Timestamp("2021-04-30")].copy()
    return df

# --------------------- Dataset construction ------------------------ #
@dataclass
class WindowedSample:
    X: np.ndarray          # [N, T*F]
    y: np.ndarray          # [N]
    symbols: List[str]

def group_data_by_symbol(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    groups: Dict[str, pd.DataFrame] = {}
    for sym in df["symbol"].unique():
        sdf = df[df["symbol"] == sym].copy()
        sdf = sdf.set_index("date").sort_index()
        sdf.index = pd.DatetimeIndex(sdf.index)
        groups[sym] = sdf
    return groups

def extract_window_for_symbol(symbol_df: pd.DataFrame, asof: pd.Timestamp, T: int,
                              features: List[str] = FEATURES, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, bool]:
    global TRADING_CAL
    wdates = last_n_trading_days(asof, T, TRADING_CAL)
    if wdates is None or len(wdates) != T:
        if debug:
            print(f"    [DEBUG] 캘린더 부족: asof={asof.date()}, T={T}")
        return np.zeros((T, len(features)), np.float32), np.zeros((T,), np.float32), False

    win_df = pd.DataFrame({"date": pd.DatetimeIndex(wdates)})
    sym_reset = symbol_df.reset_index()
    cols = ["date"] + list(features)
    sym_reset = sym_reset[cols]
    selected_df = win_df.merge(sym_reset, on="date", how="left").set_index("date")

    nan_rows = selected_df.isna().any(axis=1).sum()
    if nan_rows > 0:
        if debug:
            print(f"    [DEBUG] NaN 있는 행 수: {nan_rows}/{len(selected_df)}")
        return np.zeros((T, len(features)), np.float32), np.zeros((T,), np.float32), False

    X = selected_df.values.astype(np.float32)
    returns = selected_df["log_return"].values.astype(np.float32) if "log_return" in features else selected_df.iloc[:, 0].values.astype(np.float32)
    return X, returns, True

def build_window_matrix(df: pd.DataFrame, symbols: List[str], asof: pd.Timestamp, T: int,
                        features: List[str] = FEATURES,
                        symbol_groups: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    if symbol_groups is None:
        symbol_groups = group_data_by_symbol(df)

    X_list, returns_list, mask, kept = [], [], [], []
    for i, sym in enumerate(symbols):
        if sym not in symbol_groups:
            mask.append(False)
            X_list.append(np.zeros((T, len(features)), np.float32))
            returns_list.append(np.zeros((T,), np.float32))
            continue
        sdf = symbol_groups[sym]
        X_sym, r_sym, valid = extract_window_for_symbol(sdf, asof, T, features, debug=(i == 0))
        if not valid:
            mask.append(False)
            X_list.append(np.zeros((T, len(features)), np.float32))
            returns_list.append(np.zeros((T,), np.float32))
            continue
        kept.append(sym)
        mask.append(True)
        X_list.append(X_sym)
        returns_list.append(r_sym)

    X = np.stack(X_list, axis=0)               # [N, T, F]
    X = X.reshape(X.shape[0], -1)              # [N, T*F]
    returns_array = np.stack(returns_list, 0)  # [N, T]
    return X, np.array(mask, bool), kept, returns_array

def build_label_vector(df: pd.DataFrame, symbols: List[str], asof: pd.Timestamp) -> np.ndarray:
    future_date = asof + pd.tseries.offsets.BDay(20)
    base = (
        df[df["date"] == asof][["symbol", "close"]].set_index("symbol").reindex(symbols)["close"]
    )
    fut = (
        df[df["date"] == future_date][["symbol", "close"]].set_index("symbol").reindex(symbols)["close"]
    )
    rel = (fut / base - 1.0).astype(float)
    y = np.zeros(len(symbols), dtype=np.int64)
    y[(rel >= 0.05) & (rel < 0.10)] = 1
    y[rel >= 0.10] = 2
    return y

# --------------------- GAT + GRU model (node classification) -------- #
class TemporalGATGRU(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.2)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index):
        # x: [N, F], edge_index: [2, E]
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)        # [N, H]
        # 시간축은 이미 X에서 평균해 왔으므로, GRU에 1-step 시퀀스로 흘려보냅니다.
        x = x.unsqueeze(1)                  # [N, 1, H]
        x, _ = self.gru(x)                  # [N, 1, H]
        x = self.fc(x[:, -1, :])            # [N, 3]  <-- 노드별 로짓
        return x

class TemporalGATWrapper:
    """Sklearn-like wrapper. Expects flattened X=[N, T*F] and returns=[N, T].
       Internally builds a single-graph with N nodes. Node features = mean over time of F.
    """
    def __init__(self, T=20, hidden_dim=64, num_layers=2, num_heads=4,
                 lr=1e-3, epochs=50, corr_threshold=0.3,
                 k_neighbors=5, mutual_topk=False):   # 👈 추가
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.lr = lr
        self.epochs = epochs
        self.corr_threshold = corr_threshold
        self.k_neighbors = k_neighbors      # 👈 추가
        self.mutual_topk = mutual_topk      # 👈 추가
        self.model = None
        self.device = DEVICE
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.num_features_ = None

    # ---- graph building ---- #
    @staticmethod
    def _corr_edges_from_returns(returns: np.ndarray, thr: float) -> np.ndarray:
        """returns: [N, T]; build undirected edge_index (2, E) for |corr| >= thr (i<j)."""
        if returns.shape[0] == 0:
            return np.zeros((2, 0), dtype=np.int64)
        R = returns.astype(np.float32)
        R = R - R.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(R, axis=1, keepdims=True)
        denom = denom * denom.T + 1e-8
        corr = (R @ R.T) / denom
        np.fill_diagonal(corr, 0.0)
        ii, jj = np.where(np.abs(corr) >= thr)
        mask = ii < jj
        ii, jj = ii[mask], jj[mask]
        if ii.size == 0:
            return np.zeros((2, 0), dtype=np.int64)
        edges = np.vstack([np.concatenate([ii, jj]), np.concatenate([jj, ii])])  # make symmetric
        return edges.astype(np.int64)

    @staticmethod
    def _node_features_from_X(X: np.ndarray, T: int, Fdim: int) -> np.ndarray:
        X3 = X.reshape(X.shape[0], T, Fdim)
        feat_mean = X3.mean(axis=1)
        feat_std = X3.std(axis=1)
        feat_last = X3[:, -1, :]
        return np.concatenate([feat_mean, feat_std, feat_last], axis=1).astype(np.float32)

    @staticmethod
    def _topk_edges_from_returns(returns: np.ndarray, k: int = 5, mutual: bool = False) -> np.ndarray:
        """
        returns: [N, T]  (학습 윈도우 수익률)
        각 노드별 |corr| 기준 상위 k 이웃을 뽑아 대칭화(symmetric)한 edge_index 반환 (2, E).
        mutual=True이면 '서로의 상위 k'인 엣지만 유지(더 정밀, 더 희소).
        """
        N = returns.shape[0]
        if N <= 1 or k <= 0:
            return np.zeros((2, 0), dtype=np.int64)

        R = returns.astype(np.float32)
        R = R - R.mean(axis=1, keepdims=True)
        denom = (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
        C = (R @ R.T) / (denom * denom.T)  # 상관계수 행렬
        np.fill_diagonal(C, -np.inf)  # 자기 자신 제외

        # 각 노드별 top-k 인덱스 (절댓값 기준)
        k_eff = min(k, max(1, N - 1))
        idxs = np.argpartition(-np.abs(C), kth=k_eff - 1, axis=1)[:, :k_eff]

        edges = set()
        if mutual:
            # 상호 top-k만 허용
            for i in range(N):
                nbrs = idxs[i]
                for j in nbrs:
                    # i in topk(j)?
                    if i in idxs[j]:
                        edges.add((i, j));
                        edges.add((j, i))
        else:
            for i in range(N):
                for j in idxs[i]:
                    edges.add((i, j));
                    edges.add((j, i))

        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        ii, jj = zip(*edges)
        return np.vstack([np.array(ii, dtype=np.int64),
                          np.array(jj, dtype=np.int64)])

    # ---- API ---- #
    def fit(self, X_flat: np.ndarray, y: np.ndarray, returns_data: np.ndarray) -> "TemporalGATWrapper":
        assert X_flat.ndim == 2 and returns_data.ndim == 2
        N, TF = X_flat.shape
        T = self.T
        Fdim = TF // T

        # ✨노드 피처 먼저 생성
        x_np = self._node_features_from_X(X_flat, T, Fdim)  # shape: [N, D_feat]
        D_feat = x_np.shape[1]  # ← 실제 피처 차원 자동 계산
        self.num_features_ = D_feat

        # 엣지 생성 (top-k 또는 threshold)
        edge_index_np = self._topk_edges_from_returns(returns_data, k=self.k_neighbors, mutual=self.mutual_topk)
        if edge_index_np.shape[1] == 0:
            if N >= 2:
                ii = np.arange(N - 1, dtype=np.int64);
                jj = ii + 1
                edge_index_np = np.vstack([np.concatenate([ii, jj]), np.concatenate([jj, ii])])
            else:
                edge_index_np = np.zeros((2, 0), dtype=np.int64)

        # Torch 텐서화
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        # ✨여기서 모델 in_channels = D_feat 로 생성
        self.model = TemporalGATGRU(num_features=D_feat,
                                    hidden_dim=self.hidden_dim,
                                    num_layers=self.num_layers,
                                    num_heads=self.num_heads).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        crit = nn.CrossEntropyLoss()  # (불균형이면 가중치 적용)

        self.model.train()
        for ep in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == "cuda")):
                logits = self.model(x, edge_index)  # [N, 3]
                loss = crit(logits, y_t)
            if self.device.type == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()
            # (옵션) 주기 평가 로그
        return self

    @torch.no_grad()
    def predict_proba(self, X_flat: np.ndarray, returns_data: np.ndarray) -> np.ndarray:
        assert self.model is not None
        N, TF = X_flat.shape
        T = self.T
        Fdim = TF // T

        # ✨훈련 때와 동일한 방법으로 노드 피처 생성
        x_np = self._node_features_from_X(X_flat, T, Fdim)  # [N, D_feat]
        D_feat = x_np.shape[1]

        # 혹시 저장/로드 등으로 self.num_features_가 다르면, 한 번 체크(선택)
        if getattr(self, "num_features_", None) != D_feat:
            # 입력 피처 차원이 바뀌었다면(예: 파이프라인 변경), 로드된 모델과 안 맞습니다.
            # 이 경우는 재학습이 필요하지만, 동일 파이프라인이라면 정상적으로 일치할 겁니다.
            pass

        edge_index_np = self._topk_edges_from_returns(returns_data, k=self.k_neighbors, mutual=self.mutual_topk)
        if edge_index_np.shape[1] == 0:
            if N >= 2:
                ii = np.arange(N - 1, dtype=np.int64);
                jj = ii + 1
                edge_index_np = np.vstack([np.concatenate([ii, jj]), np.concatenate([jj, ii])])
            else:
                edge_index_np = np.zeros((2, 0), dtype=np.int64)

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == "cuda")):
            logits = self.model(x, edge_index)  # [N, 3]
            prob = F.softmax(logits, dim=1)
        return prob.detach().cpu().numpy()

# --------------------- Training / Predict orchestration ------------- #
def make_model(T: int, seed: int) -> TemporalGATWrapper:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return TemporalGATWrapper(T=T, hidden_dim=64, num_layers=2, num_heads=4, lr=1e-3, epochs=50, corr_threshold=0.3)

def train_month_end_models(df: pd.DataFrame, asof: pd.Timestamp, save_dir: str, model_idx: int,
                           n_models: int = 5) -> Dict[int, List[str]]:
    """Train n_models seeds for each T in WINDOWS and save all. Returns map T->list of paths."""
    ensure_dirs()
    os.makedirs(save_dir, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())
    symbol_groups = group_data_by_symbol(df)
    saved: Dict[int, List[str]] = {T: [] for T in WINDOWS}

    for T in WINDOWS:
        print(f"  [INFO] asof 기준 T={T} 학습: {asof.date()}")
        X, mask, kept, returns = build_window_matrix(df, symbols, asof, T, FEATURES, symbol_groups)
        if not mask.any():
            print("    [WARNING] 유효한 샘플 없음")
            continue
        y_full = build_label_vector(df, symbols, asof)
        X_kept = X[mask]
        y_kept = y_full[mask]
        returns_kept = returns[mask]

        # ensemble over seeds; save ALL models
        for i in range(n_models):
            seed = (model_idx * 1000) + (i + 1)
            print(f"    [모델 {i+1}/{n_models}] seed={seed}")
            model = make_model(T=T, seed=seed)
            model.fit(X_kept, y_kept, returns_kept)
            pack = {"model": model, "asof": asof, "T": T, "seed": seed, "features": FEATURES}
            fname = f"T{T}_seed{seed}.joblib"
            fpath = os.path.join(save_dir, fname)
            joblib.dump(pack, fpath)
            saved[T].append(fpath)  # 모든 모델 경로 저장
    return saved

@torch.no_grad()
def predict_with_ensemble(df: pd.DataFrame, asof: pd.Timestamp, predict_asof: pd.Timestamp,
                          saved_map: Dict[int, List[str]]) -> Optional[np.ndarray]:
    """Load all models for each T and ensemble their predictions."""
    symbols = sorted(df["symbol"].unique().tolist())
    all_probas: List[np.ndarray] = []

    for T in WINDOWS:
        model_paths = saved_map.get(T, [])
        if not model_paths:
            continue
        
        # 각 T에 대해 모든 모델의 예측값 수집
        probas_for_T = []
        for model_path in model_paths:
            if not os.path.exists(model_path):
                continue
            pack = joblib.load(model_path, mmap_mode=None)
            model: TemporalGATWrapper = pack["model"]
            # map to current device and eval
            model.device = DEVICE
            if model.model is not None:
                model.model.to(DEVICE)
            
            # build inputs for this T
            X_T, mask_T, kept_T, returns_T = build_window_matrix(df, symbols, predict_asof, T, FEATURES)
            if not mask_T.any():
                continue
            proba_kept = model.predict_proba(X_T[mask_T], returns_T[mask_T])
            # scatter back to full N with NaNs
            proba_full = np.full((len(symbols), 3), np.nan, dtype=np.float32)
            proba_full[mask_T] = proba_kept
            probas_for_T.append(proba_full)
        
        # 해당 T의 모든 모델 예측값 평균
        if probas_for_T:
            avg_for_T = np.nanmean(np.stack(probas_for_T, axis=0), axis=0)
            all_probas.append(avg_for_T)

    if not all_probas:
        return None

    # T=20, 40, 60의 평균을 다시 평균
    final_avg = np.nanmean(np.stack(all_probas, axis=0), axis=0)  # [N,3]
    pred = np.nanargmax(final_avg, axis=1)
    return pred


def _topk_edges_from_returns(returns: np.ndarray, k: int = 5) -> np.ndarray:
    # returns: [N,T]
    N = returns.shape[0]
    if N <= 1: return np.zeros((2,0), np.int64)
    R = returns.astype(np.float32)
    R = R - R.mean(axis=1, keepdims=True)
    denom = (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    corr = (R @ R.T) / (denom * denom.T)
    np.fill_diagonal(corr, -np.inf)

    ii_list, jj_list = [], []
    for i in range(N):
        idx = np.argpartition(-np.abs(corr[i]), kth=min(k, N-1)-1)[:k]
        idx = idx[idx != i]
        for j in idx:
            ii_list.append(i); jj_list.append(j)
            ii_list.append(j); jj_list.append(i)   # make symmetric
    if not ii_list: return np.zeros((2,0), np.int64)
    return np.vstack([np.array(ii_list), np.array(jj_list)]).astype(np.int64)


# --------------------- Rolling train & predict ---------------------- #
def train_and_predict_ensemble(df: pd.DataFrame, n_models: int = 5) -> None:
    global TRADING_CAL
    if TRADING_CAL is None or len(TRADING_CAL) == 0:
        set_trading_calendar_from_df(df)
        print(f"[INFO] Trading calendar initialized: {TRADING_CAL[0].date()} ~ {TRADING_CAL[-1].date()} ({len(TRADING_CAL)} days)")

    month_ends = get_month_ends(TRADING_CAL)
    if len(month_ends) < 2:
        raise RuntimeError("Not enough month-ends in calendar")

    print("\n" + "=" * 80)
    print("월말 영업일 기준 반복 학습 및 예측 시작")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # results 폴더에 시각 기반 제목으로 저장
    results_dir = os.path.join("results", run_tag)
    os.makedirs(results_dir, exist_ok=True)
    
    pred_dir = os.path.join(results_dir, "predictions")
    model_dir = os.path.join(results_dir, "models")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"결과 저장 디렉토리: {results_dir}")
    print("=" * 80)

    print(f"총 {len(month_ends)-1}개 월말 영업일에서 학습/예측 진행")
    print(f"학습 기간: {month_ends[0]} ~ {month_ends[-2]}")
    print(f"예측 기간: {month_ends[1]} ~ {month_ends[-1]}")

    symbols = sorted(df["symbol"].unique().tolist())
    pred_mat = []
    idx_dates = []

    for i in range(len(month_ends) - 1):
        train_asof = pd.Timestamp(month_ends[i])
        predict_asof = pd.Timestamp(month_ends[i + 1])
        print(f"\n[{i+1}/{len(month_ends)-1}] 학습: {train_asof.date()}, 예측: {predict_asof.date()}\n" + "-" * 80)

        # models 폴더가 아닌 results/{run_tag}/models/{날짜} 구조로 저장
        date_dir = os.path.join(model_dir, train_asof.strftime("%Y%m%d"))
        saved = train_month_end_models(df, train_asof, save_dir=date_dir, model_idx=(i+1), n_models=n_models)
        pred = predict_with_ensemble(df, train_asof, predict_asof, saved)
        if pred is None:
            print("  [경고] 유효한 예측 없음")
            pred = np.full(len(symbols), np.nan)
        pred_mat.append(pred)
        idx_dates.append(predict_asof)

    pred_df = pd.DataFrame(np.array(pred_mat), index=pd.DatetimeIndex(idx_dates), columns=symbols)
    xlsx_path = os.path.join(pred_dir, f"pred_matrix_{run_tag}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "predictions"
    for r in dataframe_to_rows(pred_df.reset_index().rename(columns={"index": "asof"}), index=False, header=True):
        ws.append(r)
    wb.save(xlsx_path)
    print(f"[완료] 예측 결과 저장: {xlsx_path}")

# --------------------- CLI ----------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH, help="Excel data path (preprocessed or raw)")
    parser.add_argument("--mode", choices=["auto", "train", "predict"], default="auto")
    parser.add_argument("--start", default=None, help="(optional) min month-end YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="(optional) max month-end YYYY-MM-DD")
    parser.add_argument("--n-models", type=int, default=5, help="ensemble models per T")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    df = load_and_engineer(args.data)
    set_trading_calendar_from_df(df)

    # allow overriding epochs globally
    global WINDOWS
    # If you want to limit WINDOWS for quick tests, do it here.

    # Run full rolling process (train & predict)
    train_and_predict_ensemble(df, n_models=args.n_models)

if __name__ == "__main__":
    main()
