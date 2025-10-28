"""
Temporal GAT-GRU Monthly Ensemble (GPU-enabled, top-k edges, node classification)
-------------------------------------------------------------------------------
- WINDOWS: T in {20, 40, 60}; classes: 2 (>= +10%), 1 (>= +5% & < +10%), 0 (else)
- Train at each month-end (asof), save model per T and seed
- Predict at the next month-end, export per-symbol prediction matrix to Excel

This build adds:
- CUDA/AMP, periodic loss/acc logs
- top-k per-node edges (+ mutual-topk option)
- Residual + LayerNorm in GAT block (oversmoothing 완화)
- DropEdge (훈련시만, 과평활/과적합 완화)
- Class weights (불균형 대응)
- CLI로 주요 하이퍼파라미터 튜닝 노출
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from os.path import abspath, getsize
from typing import Dict, List, Optional, Tuple

import joblib  # optional; not used when SAVE_TORCH_ONLY = True
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# ============================== CUDA / Device =============================== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x only (safe no-op otherwise)
except Exception:
    pass

torch.backends.cudnn.benchmark = True

# ============================== Global Config ============================== #
DATA_PATH = "preprocessed_data.xlsx"
MODEL_DIR = "models"
RESULT_DIR = "results"
WINDOWS = [20, 40, 60]
FEATURES = ["log_return", "vol_z", "volatility20", "mom20"]
DEFAULT_EPOCHS = 50  # overridden by --epochs

# Graph defaults (overridable by CLI)
GLOBAL_K_NEIGHBORS = 5
GLOBAL_MUTUAL_TOPK = False
GLOBAL_LOG_EVERY = 10
GLOBAL_DROP_EDGE = 0.15  # DropEdge ratio during training (0~1)

# GAT/GRU defaults (overridable by CLI)
GAT_HIDDEN = 64
GAT_HEADS = 4
GAT_LAYERS = 2
GAT_DROPOUT = 0.3
NEG_SLOPE = 0.2
GRU_LAYERS = 1
GRU_DROPOUT = 0.0

# Save options
SAVE_TORCH_ONLY = True  # True: save .pt + .json; False: pickle wrapper via joblib

# Trading calendar (derived from input Excel)
TRADING_CAL: Optional[pd.DatetimeIndex] = None

# ================================ Utilities ================================ #
def ensure_dirs() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def _ensure_datetime_series(s: pd.Series) -> pd.Series:
    """Convert Excel 'date' safely to naive normalized timestamps.
    Handles strings, timestamps, tz-aware, and Excel serial numbers.
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
    return ends.tolist()

# =============================== IO / Loading ============================== #
def load_and_engineer(data_path: str) -> pd.DataFrame:
    """Load Excel and return long-format dataframe with engineered features.
    Supports either:
      (1) preprocessed long-form (sheet "data" with columns: date, symbol, close, volume, ...), or
      (2) wide matrices with sheets "price" and "volume" (first column is date).
    """
    try:
        xl = pd.ExcelFile(data_path)
        if "data" in xl.sheet_names:
            print(f"[INFO] 전처리 데이터 로드: {data_path} (sheet='data')")
            df = pd.read_excel(data_path, sheet_name="data")
            req = {"date", "symbol", "close", "volume"}
            missing = req - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            df["date"] = _ensure_datetime_series(df["date"])  # normalize
            df["close"] = df["close"].fillna(0)
            df["volume"] = df["volume"].fillna(0)
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        else:
            print(f"[INFO] 원본 매트릭스 로드: {data_path} (sheets='price','volume')")
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
    df = df.dropna(subset=FEATURES)
    return df

# =========================== Windowed dataset ============================= #
@dataclass
class WindowedSample:
    X: np.ndarray          # [N, T*F]
    y: np.ndarray          # [N]
    symbols: List[str]


def group_data_by_symbol(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in df["symbol"].unique():
        sdf = df[df["symbol"] == sym].copy()
        sdf = sdf.set_index("date").sort_index()
        sdf.index = pd.DatetimeIndex(sdf.index)
        out[sym] = sdf
    return out


def extract_window_for_symbol(symbol_df: pd.DataFrame, asof: pd.Timestamp, T: int,
                              features: List[str] = FEATURES,
                              debug: bool = False) -> Tuple[np.ndarray, np.ndarray, bool]:
    global TRADING_CAL
    wdates = last_n_trading_days(asof, T, TRADING_CAL)
    if wdates is None or len(wdates) != T:
        if debug:
            print(f"    [DEBUG] 캘린더 부족: asof={pd.Timestamp(asof).date()}, T={T}")
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
    """3-class label by +20 trading days forward return from asof.
    If future date not available, label defaults to 0.
    """
    global TRADING_CAL
    y = np.zeros(len(symbols), dtype=np.int64)
    if TRADING_CAL is None or len(TRADING_CAL) == 0:
        return y
    asof = snap_to_calendar(asof, TRADING_CAL)
    if asof is None:
        return y
    pos = TRADING_CAL.get_indexer([asof])[0]
    fut_pos = pos + 20
    if fut_pos >= len(TRADING_CAL):
        return y
    future_date = TRADING_CAL[fut_pos]

    base = (
        df[df["date"] == asof][["symbol", "close"]].set_index("symbol").reindex(symbols)["close"]
    )
    fut = (
        df[df["date"] == future_date][["symbol", "close"]].set_index("symbol").reindex(symbols)["close"]
    )
    rel = (fut / base - 1.0).astype(float)
    y[(rel >= 0.05) & (rel < 0.10)] = 1
    y[rel >= 0.10] = 2
    return y

# ============================== Model Layers =============================== #
class TemporalGATGRU(nn.Module):
    """Node classification GAT + GRU head with residual & norm to mitigate oversmoothing.
    Input features are node-level vectors (time-aggregated over T-window).
    Output is [N, 3] logits (per node).
    """
    def __init__(self, num_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 gat_dropout: float = 0.3,
                 neg_slope: float = 0.2,
                 gru_layers: int = 1,
                 gru_dropout: float = 0.0):
        super().__init__()
        self.neg_slope = float(neg_slope)
        self.gat_dropout = float(gat_dropout)

        # Residual projection from input to hidden
        self.lin_in = nn.Linear(num_features, hidden_dim)

        # Two GAT layers
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads,
                            dropout=gat_dropout, negative_slope=neg_slope)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1,
                            dropout=gat_dropout, negative_slope=neg_slope)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # GRU head
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers,
                          dropout=(gru_dropout if gru_layers > 1 else 0.0), batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index):
        # x: [N, F], edge_index: [2, E]
        # GAT block 1
        h = F.leaky_relu(self.gat1(x, edge_index), negative_slope=self.neg_slope)
        h = F.dropout(h, p=self.gat_dropout, training=self.training)
        # GAT block 2
        h = self.gat2(h, edge_index)
        h = F.leaky_relu(h, negative_slope=self.neg_slope)
        h = self.norm2(h)
        # Residual from input
        h = h + self.lin_in(x)
        # GRU head (+ fake temporal dim)
        h = h.unsqueeze(1)
        h, _ = self.gru(h)
        return self.fc(h[:, -1, :])


class TemporalGATWrapper:
    def __init__(self, T=20, hidden_dim=64, num_layers=2, num_heads=4,
                 lr=1e-3, epochs=50, corr_threshold=0.3,
                 k_neighbors=5, mutual_topk=False,
                 log_every=10,
                 gat_dropout=0.3, neg_slope=0.2,
                 gru_layers=1, gru_dropout=0.0,
                 drop_edge_rate=0.15):
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.lr = lr
        self.epochs = epochs
        self.corr_threshold = corr_threshold
        self.k_neighbors = k_neighbors
        self.mutual_topk = mutual_topk
        self.log_every = log_every
        self.gat_dropout = gat_dropout
        self.neg_slope = neg_slope
        self.gru_layers = gru_layers
        self.gru_dropout = gru_dropout
        self.drop_edge_rate = drop_edge_rate

        self.device = DEVICE
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.model: Optional[TemporalGATGRU] = None
        self.num_features_: Optional[int] = None

    # --------- Edge building (per-node top-k) --------- #
    @staticmethod
    def _topk_edges_from_returns(returns: np.ndarray, k: int = 5, mutual: bool = False) -> np.ndarray:
        """returns: [N, T] → |corr| top-k neighbors per node → symmetric edge_index (2,E)."""
        N = returns.shape[0]
        if N <= 1 or k <= 0:
            return np.zeros((2, 0), dtype=np.int64)
        R = returns.astype(np.float32)
        R = R - R.mean(axis=1, keepdims=True)
        denom = (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
        C = (R @ R.T) / (denom * denom.T)
        np.fill_diagonal(C, -np.inf)
        k_eff = min(k, max(1, N - 1))
        idxs = np.argpartition(-np.abs(C), kth=k_eff - 1, axis=1)[:, :k_eff]
        edges = set()
        if mutual:
            for i in range(N):
                nbrs = idxs[i]
                for j in nbrs:
                    if i in idxs[j]:
                        edges.add((i, j)); edges.add((j, i))
        else:
            for i in range(N):
                for j in idxs[i]:
                    edges.add((i, j)); edges.add((j, i))
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        ii, jj = zip(*edges)
        return np.vstack([np.array(ii, dtype=np.int64), np.array(jj, dtype=np.int64)])

    # --------- Node features from windowed X --------- #
    @staticmethod
    def _node_features_from_X(X: np.ndarray, T: int, Fdim: int) -> np.ndarray:
        """Aggregate [N, T*F] → [N, D_feat] using [mean, std, last] over time."""
        X3 = X.reshape(X.shape[0], T, Fdim)
        feat_mean = X3.mean(axis=1)
        feat_std = X3.std(axis=1)
        feat_last = X3[:, -1, :]
        return np.concatenate([feat_mean, feat_std, feat_last], axis=1).astype(np.float32)

    # --------------------------- API --------------------------- #
    def fit(self, X_flat: np.ndarray, y: np.ndarray, returns_data: np.ndarray) -> "TemporalGATWrapper":
        assert X_flat.ndim == 2 and returns_data.ndim == 2
        N, TF = X_flat.shape
        T = self.T
        Fdim = TF // T

        # Build node features and edges
        x_np = self._node_features_from_X(X_flat, T, Fdim)   # [N, D_feat]
        D_feat = x_np.shape[1]
        self.num_features_ = D_feat
        edge_index_np = self._topk_edges_from_returns(returns_data, k=self.k_neighbors, mutual=self.mutual_topk)
        if edge_index_np.shape[1] == 0:
            if N >= 2:
                ii = np.arange(N - 1, dtype=np.int64); jj = ii + 1
                edge_index_np = np.vstack([np.concatenate([ii, jj]), np.concatenate([jj, ii])])
            else:
                edge_index_np = np.zeros((2, 0), dtype=np.int64)
        print(f"    [GRAPH] nodes={N}, edges={edge_index_np.shape[1]}, D_feat={D_feat}, k={self.k_neighbors}, mutual={self.mutual_topk}")

        # Torch tensors
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        # Model/optim
        self.model = TemporalGATGRU(num_features=D_feat,
                                    hidden_dim=self.hidden_dim,
                                    num_layers=self.num_layers,
                                    num_heads=self.num_heads,
                                    gat_dropout=self.gat_dropout,
                                    neg_slope=self.neg_slope,
                                    gru_layers=self.gru_layers,
                                    gru_dropout=self.gru_dropout).to(self.device)
        # opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        opt = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.lr,  # 처음엔 1e-3 그대로 써보고, 필요시 3e-4 ~ 2e-3 튜닝
            alpha=0.99,  # EMA 계수(=decay)
            momentum=0.9,  # 분류 문제에서 보통 유리
            centered=True,  # 분산 중심화 → 수렴 안정성↑ (VRAM 여유 있으면 권장)
            eps=1e-8,
            weight_decay=1e-5  # L2 규제
        )

        # Class weights (optional)
        crit: nn.Module
        try:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.array([0, 1, 2])
            w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            w = torch.tensor(w, dtype=torch.float32, device=self.device)
            crit = nn.CrossEntropyLoss(weight=w)
            print(f"    [INFO] class weights: {w.tolist()}")
        except Exception:
            crit = nn.CrossEntropyLoss()

        # Label distribution log
        uniq, cnt = np.unique(y, return_counts=True)
        print("    [INFO] train label dist:", {int(k): int(v) for k, v in zip(uniq, cnt)})

        # Local helper: DropEdge
        def _drop_edge(eidx: torch.Tensor, drop_rate: float) -> torch.Tensor:
            if eidx.numel() == 0 or drop_rate <= 0:
                return eidx
            E = eidx.size(1)
            keep = max(1, int(E * (1.0 - drop_rate)))
            idx = torch.randperm(E, device=eidx.device)[:keep]
            return eidx[:, idx]

        self.model.train()
        for ep in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            edge_train = _drop_edge(edge_index, self.drop_edge_rate) if self.model.training else edge_index
            with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == "cuda")):
                logits = self.model(x, edge_train)   # [N, 3]
                loss = crit(logits, y_t)
            if self.device.type == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                loss.backward()
                opt.step()

            # Periodic evaluation/log (after update)
            if (ep + 1) % self.log_every == 0 or ep == 0 or ep == self.epochs - 1:
                self.model.eval()
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(self.device.type == "cuda")):
                    logits_eval = self.model(x, edge_index)
                    loss_val = crit(logits_eval, y_t).item()
                    acc_val = (logits_eval.argmax(dim=1) == y_t).float().mean().item()
                self.model.train()
                print(f"    [EP {ep+1:03d}/{self.epochs}] loss={loss_val:.4f} acc={acc_val:.3f} (N={N})")
        return self

    @torch.no_grad()
    def predict_proba(self, X_flat: np.ndarray, returns_data: np.ndarray) -> np.ndarray:
        assert self.model is not None
        N, TF = X_flat.shape
        T = self.T
        Fdim = TF // T
        x_np = self._node_features_from_X(X_flat, T, Fdim)
        D_feat = x_np.shape[1]
        if self.num_features_ is not None and self.num_features_ != D_feat:
            print(f"[WARN] D_feat changed from train({self.num_features_}) to infer({D_feat}).")
        edge_index_np = self._topk_edges_from_returns(returns_data, k=self.k_neighbors, mutual=self.mutual_topk)
        if edge_index_np.shape[1] == 0:
            if N >= 2:
                ii = np.arange(N - 1, dtype=np.int64); jj = ii + 1
                edge_index_np = np.vstack([np.concatenate([ii, jj]), np.concatenate([jj, ii])])
            else:
                edge_index_np = np.zeros((2, 0), dtype=np.int64)
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == "cuda")):
            logits = self.model(x, edge_index)
            prob = F.softmax(logits, dim=1)
        return prob.detach().cpu().numpy()

# ======================== Orchestration / Saving =========================== #
def make_model(T: int, seed: int) -> TemporalGATWrapper:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return TemporalGATWrapper(
        T=T,
        hidden_dim=GAT_HIDDEN,
        num_layers=GAT_LAYERS,
        num_heads=GAT_HEADS,
        lr=1e-3,
        epochs=DEFAULT_EPOCHS,
        corr_threshold=0.3,
        k_neighbors=GLOBAL_K_NEIGHBORS,
        mutual_topk=GLOBAL_MUTUAL_TOPK,
        log_every=GLOBAL_LOG_EVERY,
        gat_dropout=GAT_DROPOUT,
        neg_slope=NEG_SLOPE,
        gru_layers=GRU_LAYERS,
        gru_dropout=GRU_DROPOUT,
        drop_edge_rate=GLOBAL_DROP_EDGE,
    )


def train_month_end_models(df: pd.DataFrame, asof: pd.Timestamp, save_dir: str, model_idx: int,
                           n_models: int = 5) -> Dict[int, Optional[str]]:
    ensure_dirs()
    os.makedirs(save_dir, exist_ok=True)
    save_dir = abspath(save_dir)
    print(f"  [INFO] 모델 저장 경로: {save_dir}")

    symbols = sorted(df["symbol"].unique().tolist())
    symbol_groups = group_data_by_symbol(df)

    saved: Dict[int, Optional[str]] = {T: None for T in WINDOWS}

    for T in WINDOWS:
        print(f"  [INFO] asof 기준 T={T} 학습: {pd.Timestamp(asof).date()}")
        X, mask, kept, returns = build_window_matrix(df, symbols, asof, T, FEATURES, symbol_groups)
        if not mask.any():
            print("    [WARNING] 유효한 샘플 없음")
            continue
        y_full = build_label_vector(df, symbols, asof)
        X_kept = X[mask]
        y_kept = y_full[mask]
        returns_kept = returns[mask]

        last_path: Optional[str] = None
        for i in range(n_models):
            seed = (model_idx * 1000) + (i + 1)
            print(f"    [모델 {i+1}/{n_models}] seed={seed}")
            model = make_model(T=T, seed=seed)
            model.fit(X_kept, y_kept, returns_kept)

            base = f"T{T}_seed{seed}"
            if SAVE_TORCH_ONLY:
                pt_path = os.path.join(save_dir, base + ".pt")
                meta_path = os.path.join(save_dir, base + ".json")
                torch.save(model.model.state_dict(), pt_path)
                meta = {
                    "asof": str(pd.Timestamp(asof).date()),
                    "T": T,
                    "seed": seed,
                    "features": FEATURES,
                    # GAT/GRU
                    "hidden_dim": model.hidden_dim,
                    "num_layers": model.num_layers,
                    "num_heads": model.num_heads,
                    "gat_dropout": model.gat_dropout,
                    "neg_slope": model.neg_slope,
                    "gru_layers": model.gru_layers,
                    "gru_dropout": model.gru_dropout,
                    # Graph
                    "corr_threshold": model.corr_threshold,
                    "k_neighbors": model.k_neighbors,
                    "mutual_topk": model.mutual_topk,
                    "drop_edge_rate": model.drop_edge_rate,
                    # Input feat dim
                    "num_features_": model.num_features_,
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"    [SAVE][pt]   {abspath(pt_path)}  ({getsize(pt_path)/1e6:.2f} MB)")
                print(f"    [SAVE][meta] {abspath(meta_path)}")
                last_path = pt_path
            else:
                pack = {"model": model, "asof": asof, "T": T, "seed": seed, "features": FEATURES}
                jb_path = os.path.join(save_dir, base + ".joblib")
                joblib.dump(pack, jb_path)
                print(f"    [SAVE][joblib] {abspath(jb_path)}  ({getsize(jb_path)/1e6:.2f} MB)")
                last_path = jb_path

        saved[T] = last_path
    return saved


@torch.no_grad()
def predict_with_ensemble(df: pd.DataFrame, asof: pd.Timestamp, predict_asof: pd.Timestamp,
                          saved_map: Dict[int, Optional[str]]) -> Optional[np.ndarray]:
    symbols = sorted(df["symbol"].unique().tolist())
    all_probas: List[np.ndarray] = []

    for T in WINDOWS:
        model_path = saved_map.get(T)
        if not model_path or not os.path.exists(model_path):
            continue

        # Load model for inference
        if SAVE_TORCH_ONLY and model_path.lower().endswith(".pt"):
            base, _ = os.path.splitext(model_path)
            meta_path = base + ".json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            w = TemporalGATWrapper(
                T=meta.get("T", T),
                hidden_dim=meta.get("hidden_dim", GAT_HIDDEN),
                num_layers=meta.get("num_layers", GAT_LAYERS),
                num_heads=meta.get("num_heads", GAT_HEADS),
                lr=1e-3, epochs=1,
                corr_threshold=meta.get("corr_threshold", 0.3),
                k_neighbors=meta.get("k_neighbors", GLOBAL_K_NEIGHBORS),
                mutual_topk=meta.get("mutual_topk", GLOBAL_MUTUAL_TOPK),
                log_every=GLOBAL_LOG_EVERY,
                gat_dropout=meta.get("gat_dropout", GAT_DROPOUT),
                neg_slope=meta.get("neg_slope", NEG_SLOPE),
                gru_layers=meta.get("gru_layers", GRU_LAYERS),
                gru_dropout=meta.get("gru_dropout", GRU_DROPOUT),
                drop_edge_rate=meta.get("drop_edge_rate", GLOBAL_DROP_EDGE),
            )
            Fdim_feat = meta.get("num_features_", None)
            if Fdim_feat is not None:
                w.num_features_ = int(Fdim_feat)
            # Build an empty internal model and load state_dict
            if w.num_features_ is None:
                w.num_features_ = 1
            w.model = TemporalGATGRU(num_features=w.num_features_, hidden_dim=w.hidden_dim,
                                      num_layers=w.num_layers, num_heads=w.num_heads,
                                      gat_dropout=w.gat_dropout, neg_slope=w.neg_slope,
                                      gru_layers=w.gru_layers, gru_dropout=w.gru_dropout).to(DEVICE)
            try:
                # PyTorch 2.4+ 권장: 안전 모드(텐서만 로드)
                state = torch.load(model_path, map_location=DEVICE, weights_only=True)
            except TypeError:
                # 구버전 호환: 인자 없는 버전 fallback
                state = torch.load(model_path, map_location=DEVICE)
            try:
                w.model.load_state_dict(state)
            except Exception as e:
                print(f"[WARN] state_dict load failed: {e}")
            w.model.eval()
            model = w
        else:
            pack = joblib.load(model_path)
            model: TemporalGATWrapper = pack["model"]
            model.device = DEVICE
            if model.model is not None:
                model.model.to(DEVICE)

        # Build inputs for prediction date
        X_T, mask_T, kept_T, returns_T = build_window_matrix(df, symbols, predict_asof, T, FEATURES)
        if not mask_T.any():
            continue
        proba_kept = model.predict_proba(X_T[mask_T], returns_T[mask_T])
        proba_full = np.full((len(symbols), 3), np.nan, dtype=np.float32)
        proba_full[mask_T] = proba_kept
        all_probas.append(proba_full)

    if not all_probas:
        return None
    avg_proba = np.nanmean(np.stack(all_probas, axis=0), axis=0)
    pred = np.nanargmax(avg_proba, axis=1)
    return pred


def train_and_predict_ensemble(df: pd.DataFrame, n_models: int = 5) -> None:
    global TRADING_CAL
    if TRADING_CAL is None or len(TRADING_CAL) == 0:
        set_trading_calendar_from_df(df)
        print(f"[INFO] Trading calendar: {TRADING_CAL[0].date()} ~ {TRADING_CAL[-1].date()} ({len(TRADING_CAL)} days)")

    month_ends = get_month_ends(TRADING_CAL)
    if len(month_ends) < 2:
        raise RuntimeError("Not enough month-ends in calendar")

    print("\n" + "=" * 80)
    print("월말 영업일 기준 반복 학습 및 예측 시작")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULT_DIR, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"결과 저장 디렉토리: {out_dir}")
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

        date_dir = os.path.join(MODEL_DIR, train_asof.strftime("%Y%m%d"))
        saved = train_month_end_models(df, train_asof, save_dir=date_dir, model_idx=(i + 1), n_models=n_models)
        pred = predict_with_ensemble(df, train_asof, predict_asof, saved)
        if pred is None:
            print("  [경고] 유효한 예측 없음")
            pred = np.full(len(symbols), np.nan)
        pred_mat.append(pred)
        idx_dates.append(predict_asof)

    # Save prediction matrix as Excel
    pred_df = pd.DataFrame(np.array(pred_mat), index=pd.DatetimeIndex(idx_dates), columns=symbols)
    xlsx_path = os.path.join(out_dir, f"pred_matrix_{run_tag}.xlsx")
    try:
        pred_df.to_excel(xlsx_path, sheet_name="predictions")
    except Exception as e:
        # fallback: CSV
        xlsx_path = os.path.join(out_dir, f"pred_matrix_{run_tag}.csv")
        pred_df.to_csv(xlsx_path)
    print(f"[완료] 예측 결과 저장: {abspath(xlsx_path)}")

# ================================== CLI =================================== #
def main():
    # global overrides FIRST
    global DEFAULT_EPOCHS, GLOBAL_K_NEIGHBORS, GLOBAL_MUTUAL_TOPK, GLOBAL_LOG_EVERY, GLOBAL_DROP_EDGE
    global GAT_HIDDEN, GAT_HEADS, GAT_LAYERS, GAT_DROPOUT, NEG_SLOPE, GRU_LAYERS, GRU_DROPOUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH, help="Excel data path (preprocessed or raw)")
    parser.add_argument("--n-models", type=int, default=5, help="# of seed models per T")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="training epochs per model")
    parser.add_argument("--k-neighbors", type=int, default=GLOBAL_K_NEIGHBORS, help="top-k neighbors per node")
    parser.add_argument("--mutual-topk", action="store_true", help="use mutual top-k (stricter, sparser edges)")
    parser.add_argument("--log-every", type=int, default=GLOBAL_LOG_EVERY, help="print loss/acc every N epochs")
    parser.add_argument("--drop-edge", type=float, default=GLOBAL_DROP_EDGE, help="DropEdge ratio during training (0~1)")
    # GAT/GRU hyperparams
    parser.add_argument("--gat-hidden", type=int, default=GAT_HIDDEN)
    parser.add_argument("--gat-heads", type=int, default=GAT_HEADS)
    parser.add_argument("--gat-layers", type=int, default=GAT_LAYERS)
    parser.add_argument("--gat-drop", type=float, default=GAT_DROPOUT)
    parser.add_argument("--neg-slope", type=float, default=NEG_SLOPE)
    parser.add_argument("--gru-layers", type=int, default=GRU_LAYERS)
    parser.add_argument("--gru-drop", type=float, default=GRU_DROPOUT)

    args = parser.parse_args()

    # propagate CLI → globals
    DEFAULT_EPOCHS      = int(args.epochs)
    GLOBAL_K_NEIGHBORS  = int(args.k_neighbors)
    GLOBAL_MUTUAL_TOPK  = bool(args.mutual_topk)
    GLOBAL_LOG_EVERY    = int(args.log_every)
    GLOBAL_DROP_EDGE    = float(args.drop_edge)

    GAT_HIDDEN   = int(args.gat_hidden)
    GAT_HEADS    = int(args.gat_heads)
    GAT_LAYERS   = int(args.gat_layers)
    GAT_DROPOUT  = float(args.gat_drop)
    NEG_SLOPE    = float(args.neg_slope)
    GRU_LAYERS   = int(args.gru_layers)
    GRU_DROPOUT  = float(args.gru_drop)

    df = load_and_engineer(args.data)
    set_trading_calendar_from_df(df)
    train_and_predict_ensemble(df, n_models=args.n_models)


if __name__ == "__main__":
    # python temporal_gat_monthly_ensemble_pipeline_GPU_full.py --data preprocessed_data.xlsx --n-models 3 --epochs 100 --k-neighbors 3 --mutual-topk --drop-edge .15 --gat-hidden 64 --gat-heads 4 --gat-layers 2 --gat-drop 0.3 --neg-slope 0.2 --gru-layers 1 --gru-drop 0.0 --log-every 10
    main()
