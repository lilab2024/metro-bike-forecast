"""
Step 4 - Station-level hourly forecasting.

Models are trained on the full station graph but the loss and the reported
metrics are restricted to one target station at a time, which is what makes
the comparison across stations with different coupling strength meaningful.
Three architectures are compared for each station and each target flow:

    MT-GCN-Attn  MBC-GCN with a learnable time embedding and temporal attention
    ST-GCN-Attn  the same backbone without the separate time-embedding branch
    LSTM         station-wise sequential baseline

Reproduces Table 7 and Figure 8 for the four stations discussed in the paper.

The script has two modes:

    --mode train   Optuna search, final training, metrics, saved checkpoints
    --mode plot    reload saved checkpoints and redraw the comparison figures
                   without retraining

Usage
-----
    python 04_forecast_station_level.py --mode train \
        --hourly-csv outputs/panel/200m/20days_data_1h.csv \
        --out-dir outputs/station_level --nodes 95 144 239 298

    python 04_forecast_station_level.py --mode plot \
        --hourly-csv outputs/panel/200m/20days_data_1h.csv \
        --out-dir outputs/station_level
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyproj import Transformer
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAMES = ["MT-GCN-Attn", "ST-GCN-Attn", "LSTM"]
TARGETS = {"ride_in": 0, "ride_out": 1}

HOURLY_RENAME = {
    "ride_in_1h": "ride_in",
    "ride_out_1h": "ride_out",
    "FINT_IN_1h": "FINT_IN",
    "FINT_OUT_1h": "FINT_OUT",
}


class Config:
    STATION_COL = "station__stationnames"
    LON_COL = "station__lon"
    LAT_COL = "station__lat"
    TIME_COL_CANDIDATES = ["FDT_STARTTIME", "hour", "time", "datetime", "start_time"]

    FEAT_COLS = [
        "ride_in",
        "ride_out",
        "FINT_IN",
        "FINT_OUT",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    # Gaussian kernel parameters in kilometres.
    THRESHOLD_KM = 2.0
    SIGMA = 1.0

    BATCH_SIZE = 32
    NUM_EPOCHS = 60
    PATIENCE = 8
    GRAD_CLIP = 5.0

    SEQ_LENGTH = 12
    GCN_OUTPUT_DIM = 80
    NUM_GCN_LAYERS = 2
    TIME_FEAT_DIM = 32
    LSTM_HIDDEN = 160
    CHEB_K = 3
    DROPOUT = 0.15

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    N_TRIALS = 20
    OPTUNA_EPOCHS = 30
    RANDOM_SEED = 20260301

    NUM_NODES = None
    INPUT_FEATS = None


def get_device() -> torch.device:
    return torch.device("cpu")


def clear_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def copy_config(base: Config):
    """Shallow copy so that each Optuna trial can mutate its own settings."""

    class Copied:
        pass

    copied = Copied()
    for attr in dir(base):
        if attr.startswith("__") or callable(getattr(base, attr)):
            continue
        setattr(copied, attr, getattr(base, attr))
    return copied


# --------------------------------------------------------------------------
# Feature scaling
# --------------------------------------------------------------------------
class FeatureProcessor:
    """Separate standardisation for the bike channels and the metro channels."""

    def __init__(self):
        self.bike_scaler = StandardScaler()
        self.metro_scaler = StandardScaler()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        steps, nodes, _ = data.shape
        scaled = data.copy()
        scaled[:, :, 0:2] = self.bike_scaler.fit_transform(
            data[:, :, 0:2].reshape(-1, 2)
        ).reshape(steps, nodes, 2)
        scaled[:, :, 2:4] = self.metro_scaler.fit_transform(
            data[:, :, 2:4].reshape(-1, 2)
        ).reshape(steps, nodes, 2)
        return scaled

    def inverse_transform_target(self, scaled: np.ndarray, target_index: int) -> np.ndarray:
        shape = scaled.shape
        flat = scaled.flatten()
        column = target_index if target_index < 2 else target_index - 2
        scaler = self.bike_scaler if target_index < 2 else self.metro_scaler
        dummy = np.zeros((len(flat), 2))
        dummy[:, column] = flat
        return scaler.inverse_transform(dummy)[:, column].reshape(shape)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def parse_time_column(df: pd.DataFrame, candidates: list[str]):
    for name in candidates:
        if name in df.columns:
            timestamps = pd.to_datetime(df[name], errors="coerce")
            if timestamps.notna().mean() > 0.9:
                return timestamps
    raise ValueError("No parseable time column found in the hourly panel.")


def load_data_and_graph(csv_path: Path, cfg: Config):
    """Return the scaled (T, N, F) tensor, the normalised adjacency and the scaler."""
    df = pd.read_csv(csv_path).rename(columns=HOURLY_RENAME)
    df["FDT_STARTTIME"] = parse_time_column(df, cfg.TIME_COL_CANDIDATES)

    station_info = (
        df[[cfg.STATION_COL, cfg.LON_COL, cfg.LAT_COL]].dropna().drop_duplicates()
    )
    cfg.NUM_NODES = len(station_info)
    cfg.INPUT_FEATS = len(cfg.FEAT_COLS)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(
        station_info[cfg.LON_COL].values, station_info[cfg.LAT_COL].values
    )
    coords = np.vstack([xs, ys]).T
    dist_km = cdist(coords, coords) / 1000.0
    adj = np.exp(-(dist_km**2) / (2 * cfg.SIGMA**2))
    adj[dist_km > cfg.THRESHOLD_KM] = 0.0

    a_tilde = adj + np.eye(cfg.NUM_NODES)
    d_inv = np.diag(1.0 / np.sqrt(a_tilde.sum(axis=1) + 1e-8))
    adjacency = torch.FloatTensor(d_inv @ a_tilde @ d_inv)

    df["hour"] = df["FDT_STARTTIME"].dt.hour
    df["dow"] = df["FDT_STARTTIME"].dt.dayofweek
    for column, period in [("hour", 24), ("dow", 7)]:
        df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / period)
        df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / period)

    df["node_id"] = df[cfg.STATION_COL].astype("category").cat.codes
    df = df.sort_values(["FDT_STARTTIME", "node_id"])
    df["time_id"] = df["FDT_STARTTIME"].astype("category").cat.codes

    num_steps = int(df["time_id"].max()) + 1
    data = np.zeros((num_steps, cfg.NUM_NODES, cfg.INPUT_FEATS), dtype=np.float32)
    subset = df[["time_id", "node_id"] + cfg.FEAT_COLS].dropna()
    data[subset["time_id"].to_numpy(), subset["node_id"].to_numpy(), :] = subset[
        cfg.FEAT_COLS
    ].to_numpy(dtype=np.float32)

    processor = FeatureProcessor()
    return processor.fit_transform(data), adjacency, processor


class SingleTargetDataset(Dataset):
    """Predict one flow channel one step ahead for every station."""

    def __init__(self, tensor: np.ndarray, seq_len: int, target_index: int):
        self.data = tensor
        self.seq_len = seq_len
        self.target_index = target_index

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, idx: int):
        window = self.data[idx : idx + self.seq_len]
        target = self.data[idx + self.seq_len, :, self.target_index][:, None]
        return torch.from_numpy(window).float(), torch.from_numpy(target).float()


def build_loaders(tensor, seq_len, batch_size, target_index):
    """Chronological 80/20 split with warm-up history kept in the validation set."""
    split = int(tensor.shape[0] * 0.8)
    train_loader = DataLoader(
        SingleTargetDataset(tensor[:split], seq_len, target_index),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SingleTargetDataset(tensor[split - seq_len :], seq_len, target_index),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class ChebGraphConv(nn.Module):
    """Chebyshev spectral graph convolution of order K."""

    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.k = k
        self.weights = nn.Parameter(torch.empty(k, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, laplacian):
        batch, steps, nodes, channels = x.shape
        x_flat = x.reshape(batch * steps, nodes, channels)
        out = torch.matmul(x_flat, self.weights[0]) + self.bias

        if self.k > 1:
            expanded = laplacian.unsqueeze(0).expand(batch * steps, -1, -1)
            t_prev, t_curr = x_flat, x_flat
            for order in range(1, self.k):
                product = torch.bmm(expanded, t_curr)
                t_next = product if order == 1 else 2.0 * product - t_prev
                out = out + torch.matmul(t_next, self.weights[order])
                t_prev, t_curr = t_curr, t_next

        return out.reshape(batch, steps, nodes, -1)


class TemporalAttention(nn.Module):
    """Per-station softmax weighting of the historical steps."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        batch, steps, nodes, channels = x.shape
        reshaped = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        weights = F.softmax(self.attention(reshaped), dim=1)
        pooled = (reshaped * weights).sum(dim=1).reshape(batch, nodes, channels)
        return pooled, weights.reshape(batch, nodes, steps)


class STGCNBlockWithDropout(nn.Module):
    def __init__(self, in_channels, out_channels, cheb_k, dropout=0.1):
        super().__init__()
        self.temporal1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.temporal2 = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.gcn = ChebGraphConv(out_channels, out_channels, cheb_k)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, laplacian):
        h1 = self.dropout(
            torch.relu(self.temporal1(x.permute(0, 3, 2, 1))).permute(0, 3, 2, 1)
        )
        h_graph = torch.relu(self.gcn(h1, laplacian))
        h2 = self.dropout(self.temporal2(h_graph.permute(0, 3, 2, 1)).permute(0, 3, 2, 1))
        return self.norm(h1 + h2)


class MTGCNWithAttention(nn.Module):
    def __init__(self, cfg, laplacian):
        super().__init__()
        dropout = getattr(cfg, "DROPOUT", 0.1)
        self.laplacian = laplacian
        self.time_embed = nn.Sequential(
            nn.Linear(4, cfg.TIME_FEAT_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.TIME_FEAT_DIM, cfg.TIME_FEAT_DIM),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                STGCNBlockWithDropout(
                    4 + cfg.TIME_FEAT_DIM if i == 0 else cfg.GCN_OUTPUT_DIM,
                    cfg.GCN_OUTPUT_DIM,
                    cfg.CHEB_K,
                    dropout,
                )
                for i in range(cfg.NUM_GCN_LAYERS)
            ]
        )
        self.temporal_attention = TemporalAttention(cfg.GCN_OUTPUT_DIM)
        self.fc = nn.Sequential(
            nn.Linear(cfg.GCN_OUTPUT_DIM, cfg.GCN_OUTPUT_DIM // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.GCN_OUTPUT_DIM // 2, 1),
        )

    def forward(self, x):
        h = torch.cat([x[:, :, :, :4], self.time_embed(x[:, :, :, 4:])], dim=-1)
        for block in self.blocks:
            h = block(h, self.laplacian)
        h, _ = self.temporal_attention(h)
        return self.fc(h)


class AttentionSTGCNPredictor(nn.Module):
    def __init__(self, cfg, laplacian):
        super().__init__()
        dropout = getattr(cfg, "DROPOUT", 0.1)
        self.laplacian = laplacian
        self.blocks = nn.ModuleList(
            [
                STGCNBlockWithDropout(
                    cfg.INPUT_FEATS if i == 0 else cfg.GCN_OUTPUT_DIM,
                    cfg.GCN_OUTPUT_DIM,
                    cfg.CHEB_K,
                    dropout,
                )
                for i in range(cfg.NUM_GCN_LAYERS)
            ]
        )
        self.temporal_attention = TemporalAttention(cfg.GCN_OUTPUT_DIM)
        self.fc = nn.Sequential(
            nn.Linear(cfg.GCN_OUTPUT_DIM, cfg.GCN_OUTPUT_DIM // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.GCN_OUTPUT_DIM // 2, 1),
        )

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h, self.laplacian)
        h, _ = self.temporal_attention(h)
        return self.fc(h)


class LSTMPredictorSingle(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dropout = getattr(cfg, "DROPOUT", 0.1)
        self.lstm = nn.LSTM(
            cfg.INPUT_FEATS,
            cfg.LSTM_HIDDEN,
            cfg.NUM_GCN_LAYERS,
            batch_first=True,
            dropout=dropout if cfg.NUM_GCN_LAYERS > 1 else 0,
        )
        self.fc = nn.Linear(cfg.LSTM_HIDDEN, 1)

    def forward(self, x):
        batch, steps, nodes, feats = x.shape
        out, _ = self.lstm(x.reshape(batch * nodes, steps, feats))
        return self.fc(out[:, -1, :]).reshape(batch, nodes, 1)


def build_model(cfg, model_name: str, adjacency: torch.Tensor, device: torch.device):
    if model_name == "MT-GCN-Attn":
        model = MTGCNWithAttention(cfg, adjacency.to(device))
    elif model_name == "ST-GCN-Attn":
        model = AttentionSTGCNPredictor(cfg, adjacency.to(device))
    elif model_name == "LSTM":
        model = LSTMPredictorSingle(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def smooth_l1(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Quadratic for small residuals, linear for large ones."""
    diff = torch.abs(pred - target)
    return torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta).mean()


def train_one(model, train_loader, val_loader, cfg, device, target_node, silent=False):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = smooth_l1(
                model(x.to(device))[:, target_node, 0], y.to(device)[:, target_node, 0]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += smooth_l1(
                    model(x.to(device))[:, target_node, 0],
                    y.to(device)[:, target_node, 0],
                ).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val, best_state, no_improve = val_loss, {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }, 0
        else:
            no_improve += 1
            if no_improve >= cfg.PATIENCE:
                if not silent:
                    print(f"  early stop at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_validation(model, tensor, seq_len, target_index, batch_size, processor, device):
    """Inference over the validation window, returned in original flow units."""
    split = int(tensor.shape[0] * 0.8)
    loader = DataLoader(
        SingleTargetDataset(tensor[split - seq_len :], seq_len, target_index),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            trues.append(y.numpy())
            preds.append(model(x.to(device)).cpu().numpy())

    y_true = processor.inverse_transform_target(np.concatenate(trues)[:, :, 0], target_index)
    y_pred = np.clip(
        processor.inverse_transform_target(np.concatenate(preds)[:, :, 0], target_index),
        0,
        None,
    )
    return y_true, y_pred


def make_objective(model_name, base_cfg, tensor, adjacency, device, target_index,
                   target_node, processor):
    def objective(trial):
        cfg = copy_config(base_cfg)
        cfg.GCN_OUTPUT_DIM = trial.suggest_categorical("gcn_dim", [64, 80, 96])
        cfg.NUM_GCN_LAYERS = trial.suggest_int("gcn_layers", 2, 3)
        cfg.LSTM_HIDDEN = trial.suggest_categorical("lstm_hidden", [128, 160])
        cfg.LEARNING_RATE = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        cfg.WEIGHT_DECAY = trial.suggest_float("wd", 1e-7, 1e-3, log=True)
        cfg.SEQ_LENGTH = trial.suggest_categorical("seq_len", [8, 12, 16])
        cfg.DROPOUT = trial.suggest_float("dropout", 0.05, 0.25)
        cfg.BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64])
        cfg.TIME_FEAT_DIM = trial.suggest_categorical("time_feat_dim", [16, 32])
        cfg.NUM_EPOCHS = base_cfg.OPTUNA_EPOCHS

        model = None
        try:
            train_loader, val_loader = build_loaders(
                tensor, cfg.SEQ_LENGTH, cfg.BATCH_SIZE, target_index
            )
            model = build_model(cfg, model_name, adjacency, device)
            model = train_one(
                model, train_loader, val_loader, cfg, device, target_node, silent=True
            )
            y_true, y_pred = predict_validation(
                model, tensor, cfg.SEQ_LENGTH, target_index, cfg.BATCH_SIZE, processor, device
            )
            score = float(
                np.sqrt(mean_squared_error(y_true[:, target_node], y_pred[:, target_node]))
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                score = 999.0
            else:
                raise
        finally:
            del model
            clear_memory()
        return score

    return objective


def train_and_evaluate(model_name, base_cfg, tensor, adjacency, device, target_name,
                       target_index, target_node, best_params, processor, out_dir):
    """Retrain with the selected hyperparameters and record the station metrics."""
    clear_memory()
    cfg = copy_config(base_cfg)
    cfg.LEARNING_RATE = best_params.get("lr", cfg.LEARNING_RATE)
    cfg.WEIGHT_DECAY = best_params.get("wd", cfg.WEIGHT_DECAY)
    cfg.SEQ_LENGTH = best_params.get("seq_len", cfg.SEQ_LENGTH)
    cfg.GCN_OUTPUT_DIM = best_params.get("gcn_dim", cfg.GCN_OUTPUT_DIM)
    cfg.NUM_GCN_LAYERS = best_params.get("gcn_layers", cfg.NUM_GCN_LAYERS)
    cfg.LSTM_HIDDEN = best_params.get("lstm_hidden", cfg.LSTM_HIDDEN)
    cfg.DROPOUT = best_params.get("dropout", 0.15)
    cfg.BATCH_SIZE = best_params.get("batch_size", cfg.BATCH_SIZE)
    cfg.TIME_FEAT_DIM = best_params.get("time_feat_dim", cfg.TIME_FEAT_DIM)

    train_loader, val_loader = build_loaders(
        tensor, cfg.SEQ_LENGTH, cfg.BATCH_SIZE, target_index
    )
    model = build_model(cfg, model_name, adjacency, device)
    model = train_one(model, train_loader, val_loader, cfg, device, target_node)

    y_true, y_pred = predict_validation(
        model, tensor, cfg.SEQ_LENGTH, target_index, cfg.BATCH_SIZE, processor, device
    )
    metrics = {
        "MAE": float(mean_absolute_error(y_true[:, target_node], y_pred[:, target_node])),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true[:, target_node], y_pred[:, target_node]))
        ),
        "R2": float(r2_score(y_true[:, target_node], y_pred[:, target_node])),
    }
    print(
        f"[{target_name} | {model_name} | node {target_node}] "
        f"MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} R2={metrics['R2']:.4f}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{target_name}_node{target_node}"
    torch.save(model.state_dict(), out_dir / f"{stem}.pt")
    with open(out_dir / f"{stem}_config.json", "w") as handle:
        json.dump(
            {
                "model_name": model_name,
                "target_name": target_name,
                "target_node": target_node,
                "best_params": best_params,
                "metrics": metrics,
            },
            handle,
            indent=2,
        )

    del model
    clear_memory()
    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
PLOT_STYLE = {
    "Actual": dict(color="#2171b5", ls="-", lw=2.5, zorder=6, alpha=1.0),
    "MT-GCN-Attn": dict(color="#D55E00", ls="--", lw=2.0, zorder=4, alpha=0.92),
    "ST-GCN-Attn": dict(color="#009E73", ls="-.", lw=2.0, zorder=4, alpha=0.92),
    "LSTM": dict(color="#CC79A7", ls=":", lw=2.1, zorder=4, alpha=0.92),
}
# Per-station y-axis limits keep the ride-in and ride-out panels comparable.
NODE_YLIM = {95: 100, 144: 175, 239: 100, 298: 100}


def plot_node(node: int, results: dict, out_dir: Path, window: int = 80) -> None:
    """Two panels per station: observed and predicted ride-in and ride-out."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    fig.subplots_adjust(wspace=0.28)

    panels = [("ride_in", "Ride-in (Inflow)"), ("ride_out", "Ride-out (Outflow)")]
    y_max = NODE_YLIM.get(node, 200)

    for axis, (target_name, label) in zip(axes, panels):
        target_results = results.get(target_name, {})
        if not target_results:
            axis.set_visible(False)
            continue

        first = next(iter(target_results))
        y_true = target_results[first]["y_true"]
        steps = min(window, len(y_true))

        style = PLOT_STYLE["Actual"]
        axis.plot(y_true[:steps, node], label="Actual", **style)
        for model_name in MODEL_NAMES:
            if model_name not in target_results:
                continue
            axis.plot(
                target_results[model_name]["y_pred"][:steps, node],
                label=model_name,
                **PLOT_STYLE[model_name],
            )

        ticks = np.arange(0, steps, 8)
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{int(t)}h" for t in ticks], fontsize=17)
        axis.set_xlim(0, steps - 1)
        axis.set_ylim(0, y_max)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        axis.tick_params(axis="y", labelsize=17)
        axis.set_title(f"Total Flow - {label}\n(Node {node})", fontsize=22,
                       fontweight="bold", pad=10)
        axis.set_xlabel("Time (hours)", fontsize=20, labelpad=6)
        axis.set_ylabel("Flow (trips/hour)", fontsize=20, labelpad=6)
        axis.legend(loc="upper right", fontsize=16, framealpha=0.88, edgecolor="#cccccc")
        axis.grid(True, ls=":", alpha=0.45, lw=0.9)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)
        axis.spines["left"].set_linewidth(1.3)
        axis.spines["bottom"].set_linewidth(1.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        path = out_dir / f"node_{node:03d}_dual.{extension}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


def load_checkpoint(model_name, target_name, node, model_dir, cfg, adjacency, device):
    """Rebuild a saved model from its config file and checkpoint."""
    stem = f"{model_name}_{target_name}_node{node}"
    config_path = model_dir / f"{stem}_config.json"
    weights_path = model_dir / f"{stem}.pt"
    if not config_path.exists() or not weights_path.exists():
        print(f"  missing checkpoint for {stem}")
        return None, None

    with open(config_path) as handle:
        saved = json.load(handle)
    params = saved["best_params"]

    run_cfg = copy_config(cfg)
    run_cfg.GCN_OUTPUT_DIM = params.get("gcn_dim", cfg.GCN_OUTPUT_DIM)
    run_cfg.NUM_GCN_LAYERS = params.get("gcn_layers", cfg.NUM_GCN_LAYERS)
    run_cfg.LSTM_HIDDEN = params.get("lstm_hidden", cfg.LSTM_HIDDEN)
    run_cfg.TIME_FEAT_DIM = params.get("time_feat_dim", cfg.TIME_FEAT_DIM)
    run_cfg.DROPOUT = params.get("dropout", cfg.DROPOUT)

    model = build_model(run_cfg, model_name, adjacency, device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, params.get("seq_len", cfg.SEQ_LENGTH)


# --------------------------------------------------------------------------
def run_train(args) -> None:
    cfg = Config()
    cfg.N_TRIALS = args.n_trials
    device = get_device()

    print(f"Device: {device} | target nodes: {args.nodes}")
    tensor, adjacency, processor = load_data_and_graph(args.hourly_csv, cfg)
    print(f"Data tensor: {tensor.shape}")
    assert all(0 <= n < cfg.NUM_NODES for n in args.nodes), (
        f"Node index out of range [0, {cfg.NUM_NODES})"
    )

    results = {}
    total = len(TARGETS) * len(args.nodes) * len(MODEL_NAMES)
    done, started = 0, time.time()

    for target_name, target_index in TARGETS.items():
        results[target_name] = {}
        for node in args.nodes:
            results[target_name][node] = {}
            for model_name in MODEL_NAMES:
                done += 1
                print(f"\n[{done}/{total}] {target_name} | node {node} | {model_name}")

                study = optuna.create_study(
                    study_name=f"{target_name}_{model_name}_node{node}",
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
                )
                study.optimize(
                    make_objective(
                        model_name, cfg, tensor, adjacency, device,
                        target_index, node, processor,
                    ),
                    n_trials=cfg.N_TRIALS,
                    show_progress_bar=True,
                )
                print(f"  best RMSE {study.best_value:.4f} with {study.best_params}")

                results[target_name][node][model_name] = train_and_evaluate(
                    model_name, cfg, tensor, adjacency, device, target_name,
                    target_index, node, study.best_params, processor, args.out_dir,
                )

                del study
                clear_memory()
                elapsed = time.time() - started
                print(f"  elapsed {elapsed / 60:.1f} min, "
                      f"eta {elapsed / done * (total - done) / 60:.1f} min")

    print("\nSummary")
    for target_name, node_results in results.items():
        print(f"\n{target_name}")
        print(f"{'Node':<8}{'Model':<16}{'MAE':<10}{'RMSE':<10}{'R2':<10}")
        for node, model_results in node_results.items():
            for model_name, payload in model_results.items():
                m = payload["metrics"]
                print(f"{node:<8}{model_name:<16}{m['MAE']:<10.2f}"
                      f"{m['RMSE']:<10.2f}{m['R2']:<10.4f}")

    summary = {
        target: {
            str(node): {name: payload["metrics"] for name, payload in models.items()}
            for node, models in nodes.items()
        }
        for target, nodes in results.items()
    }
    with open(args.out_dir / "all_results.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    for node in args.nodes:
        plot_node(node, {t: results[t][node] for t in TARGETS}, args.out_dir / "figures")

    print(f"\nAll outputs written to {args.out_dir}")


def run_plot(args) -> None:
    """Redraw the station figures from saved checkpoints without retraining."""
    cfg = Config()
    device = get_device()
    tensor, adjacency, processor = load_data_and_graph(args.hourly_csv, cfg)

    for node in args.nodes:
        print(f"\nNode {node}")
        node_results = {}
        for target_name, target_index in TARGETS.items():
            node_results[target_name] = {}
            for model_name in MODEL_NAMES:
                model, seq_len = load_checkpoint(
                    model_name, target_name, node, args.out_dir, cfg, adjacency, device
                )
                if model is None:
                    continue
                y_true, y_pred = predict_validation(
                    model, tensor, seq_len, target_index, 64, processor, device
                )
                metrics = {
                    "MAE": float(mean_absolute_error(y_true[:, node], y_pred[:, node])),
                    "RMSE": float(
                        np.sqrt(mean_squared_error(y_true[:, node], y_pred[:, node]))
                    ),
                    "R2": float(r2_score(y_true[:, node], y_pred[:, node])),
                }
                node_results[target_name][model_name] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "metrics": metrics,
                }
                print(f"  {target_name} | {model_name}: MAE={metrics['MAE']:.2f} "
                      f"RMSE={metrics['RMSE']:.2f} R2={metrics['R2']:.4f}")
                del model
                clear_memory()

        plot_node(node, node_results, args.out_dir / "figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "plot"], default="train")
    parser.add_argument(
        "--hourly-csv",
        type=Path,
        default=Path("outputs/panel/200m/20days_data_1h.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/station_level"))
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[95, 144, 239, 298],
        help="Node indices, assigned by alphabetical station order.",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        run_train(args)
    else:
        run_plot(args)


if __name__ == "__main__":
    main()
