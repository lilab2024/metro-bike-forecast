"""
Step 3 - Network-wide (all-station) demand forecasting.

Two scenarios share this script:

Scenario A (--resolution 5min)
    System-wide forecasting on the 5-minute station-time panel written by
    `01_build_station_panel.py`. LSTM, ST-GCN and MBC-GCN are tuned with
    Optuna and compared on held-out data. Reproduces Table 6 and Figure 7.

Scenario B (--resolution 1h)
    System-wide forecasting on the hourly panel. The `--features` flag
    switches between the autoregressive setting (bike flows and calendar
    features only) and the semi-autoregressive setting (bike flows plus
    metro entries and exits). Reproduces the model rows of Table 4; the
    marginal contribution of metro information in Table 5 is obtained by
    differencing the two runs.

Both scenarios use a chronological 80/20 train/validation split, a
Gaussian-kernel spatial adjacency matrix and per-target inverse scaling
before metrics are computed.

Usage
-----
    python 03_forecast_global.py --resolution 5min \
        --panel-dir outputs/panel/100m --out-dir outputs/forecast_5min

    python 03_forecast_global.py --resolution 1h \
        --hourly-csv outputs/panel/200m/20days_data_1h.csv \
        --features semi_ar --out-dir outputs/forecast_1h_semi_ar
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
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

TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def add_cyclical_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Encode hour-of-day and day-of-week as sine/cosine pairs."""
    df["hour"] = df[time_col].dt.hour
    df["dow"] = df[time_col].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    return df


def normalise_adjacency(adj: np.ndarray) -> torch.Tensor:
    """Symmetrically normalised adjacency with self-loops, D^-1/2 (A+I) D^-1/2."""
    a_tilde = adj + np.eye(adj.shape[0])
    degree = a_tilde.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    return torch.FloatTensor(d_inv_sqrt @ a_tilde @ d_inv_sqrt)


def evaluate_flows(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE and R2 for ride-in and ride-out, pooled over stations and time."""
    metrics = {}
    for index, name in enumerate(["ride_in", "ride_out"]):
        true = y_true[..., index].reshape(-1)
        pred = y_pred[..., index].reshape(-1)
        valid = np.isfinite(true) & np.isfinite(pred)
        true, pred = true[valid], pred[valid]
        metrics[name] = {
            "MAE": float(mean_absolute_error(true, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
            "R2": float(r2_score(true, pred)) if np.std(true) > 1e-6 else float("nan"),
        }
    return metrics


# ==========================================================================
# Shared graph convolution
# ==========================================================================
class ChebGraphConv(nn.Module):
    """Chebyshev spectral graph convolution of order K (Defferrard et al., 2016)."""

    def __init__(self, in_channels: int, out_channels: int, k: int, init_gain: float = 1.0):
        super().__init__()
        self.k = k
        self.weights = nn.Parameter(torch.empty(k, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weights, gain=init_gain)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C)
        batch, steps, nodes, channels = x.shape
        x_flat = x.reshape(batch * steps, nodes, channels)

        t_prev = x_flat
        out = torch.matmul(t_prev, self.weights[0])

        if self.k > 1:
            t_curr = torch.matmul(laplacian, x_flat)
            out = out + torch.matmul(t_curr, self.weights[1])
            for order in range(2, self.k):
                t_next = 2.0 * torch.matmul(laplacian, t_curr) - t_prev
                out = out + torch.matmul(t_next, self.weights[order])
                t_prev, t_curr = t_curr, t_next

        return (out + self.bias).reshape(batch, steps, nodes, -1)


# ==========================================================================
# Scenario A: 5-minute network-wide forecasting
# ==========================================================================
class FiveMinConfig:
    STATION_COL = "station__stationnames"
    TIME_COL = "FDT_STARTTIME"
    COORD_COLS = ["station__lon", "station__lat"]
    METRO_COLS = ["FINT_IN", "FINT_OUT"]
    TARGET_COLS = ["ride_in", "ride_out"]

    START_DAY = 12
    END_DAY = 31

    SEQ_LEN = 12
    # Gaussian kernel parameters operate on raw degree distances at this scale.
    THRESHOLD = 0.015
    SIGMA = 0.005

    MAX_EPOCHS = 100
    BATCH_SIZE = 64
    PATIENCE = 10
    N_TRIALS = 50


def load_5min_panel(panel_dir: Path, cfg: FiveMinConfig):
    """Build the (T, N, F) tensor and spatial graph from the 5-minute panels."""
    frames = []
    for day in range(cfg.START_DAY, cfg.END_DAY + 1):
        path = panel_dir / f"merged_202212{day:02d}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No merged_202212*.csv found in {panel_dir}")

    # The first day defines the reference station list and node ordering.
    station_info = (
        frames[0][[cfg.STATION_COL] + cfg.COORD_COLS]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    num_nodes = len(station_info)
    station_to_id = {name: i for i, name in enumerate(station_info[cfg.STATION_COL])}

    data = pd.concat(frames, ignore_index=True)
    flow_cols = cfg.METRO_COLS + cfg.TARGET_COLS
    data[flow_cols] = data[flow_cols].fillna(0)
    data[cfg.COORD_COLS] = data[cfg.COORD_COLS].ffill().bfill()
    data[cfg.TIME_COL] = pd.to_datetime(data[cfg.TIME_COL])
    data = data[data[cfg.STATION_COL].isin(station_to_id)].copy()

    # One row per (time, station): flows are summed, coordinates averaged.
    agg = {col: "sum" for col in flow_cols}
    agg.update({col: "mean" for col in cfg.COORD_COLS})
    panel = data.groupby([cfg.TIME_COL, cfg.STATION_COL], as_index=False).agg(agg)

    panel = add_cyclical_time_features(panel, cfg.TIME_COL)
    panel = panel.sort_values([cfg.TIME_COL, cfg.STATION_COL]).reset_index(drop=True)
    panel["node_id"] = panel[cfg.STATION_COL].map(station_to_id).astype(int)

    coords = station_info[cfg.COORD_COLS].values
    dist = cdist(coords, coords, metric="euclidean")
    adj = np.exp(-(dist**2) / (2 * cfg.SIGMA**2))
    adj[dist > cfg.THRESHOLD] = 0.0
    np.fill_diagonal(adj, 0.0)
    adjacency = normalise_adjacency(adj)

    # Targets are scaled per station so that large hubs do not dominate the loss;
    # inputs share one global scaler. Cast first: the raw counts are integers and
    # cannot hold standardised values on partial assignment.
    panel[cfg.TARGET_COLS] = panel[cfg.TARGET_COLS].astype(np.float32)
    target_scalers = {}
    for node, group in panel.groupby("node_id"):
        scaler = StandardScaler()
        panel.loc[group.index, cfg.TARGET_COLS] = scaler.fit_transform(
            group[cfg.TARGET_COLS]
        )
        target_scalers[node] = scaler
    for node in range(num_nodes):
        if node not in target_scalers:
            identity = StandardScaler()
            identity.mean_ = np.zeros(2)
            identity.scale_ = np.ones(2)
            target_scalers[node] = identity

    input_cols = cfg.METRO_COLS + TIME_FEATURES
    panel[input_cols] = StandardScaler().fit_transform(panel[input_cols])

    tensor_cols = cfg.TARGET_COLS + cfg.METRO_COLS + TIME_FEATURES
    panel["time_id"] = panel[cfg.TIME_COL].rank(method="dense").astype(int) - 1
    num_steps = int(panel["time_id"].max()) + 1

    tensor = np.zeros((num_steps, num_nodes, len(tensor_cols)), dtype=np.float32)
    tensor[panel["time_id"].to_numpy(), panel["node_id"].to_numpy(), :] = panel[
        tensor_cols
    ].to_numpy(dtype=np.float32)

    return tensor, adjacency, target_scalers, station_info


class FlattenedSequenceDataset(Dataset):
    """Sequences flattened over stations, for the station-agnostic LSTM."""

    def __init__(self, tensor: np.ndarray, seq_len: int):
        self.data = tensor
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, idx: int):
        window = self.data[idx : idx + self.seq_len]
        target = self.data[idx + self.seq_len, :, :2]
        return (
            torch.FloatTensor(window.reshape(self.seq_len, -1)),
            torch.FloatTensor(target.reshape(-1)),
        )


class GraphSequenceDataset(Dataset):
    """Sequences kept in (T, N, F) shape, for the graph-based models."""

    def __init__(self, tensor: np.ndarray, seq_len: int):
        self.data = tensor
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, idx: int):
        window = self.data[idx : idx + self.seq_len]
        target = self.data[idx + self.seq_len, :, :2]
        return torch.FloatTensor(window), torch.FloatTensor(target)


class MultiNodeLSTM(nn.Module):
    """Sequential baseline: the whole network state as one feature vector."""

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, num_nodes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_nodes * 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ResidualSTGCNBlock(nn.Module):
    """Temporal convolution, Chebyshev graph convolution, residual connection."""

    def __init__(self, in_channels, out_channels, k, dropout=0.1):
        super().__init__()
        self.temporal_conv1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.temporal_conv2 = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.gcn = ChebGraphConv(out_channels, out_channels, k, init_gain=0.5)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        )

    def forward(self, x, laplacian):
        residual = x

        h = x.permute(0, 3, 2, 1)
        h = F.gelu(self.temporal_conv1(h)).permute(0, 3, 2, 1)
        h = self.dropout(self.norm1(h))

        h_graph = self.dropout(F.gelu(self.gcn(h, laplacian)))

        h_out = self.temporal_conv2(h_graph.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        return self.norm2(h_out + residual)


class TemporalAttentionPool(nn.Module):
    """Softmax-weighted pooling over the observation window."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        weights = F.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


class STGCNPredictor(nn.Module):
    """Spatial baseline: all channels enter one shared spatiotemporal backbone."""

    def __init__(self, in_feat_dim, hidden_dim, num_layers, k, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_feat_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualSTGCNBlock(hidden_dim, hidden_dim, k, dropout) for _ in range(num_layers)]
        )
        self.attention = TemporalAttentionPool(hidden_dim)
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, laplacian):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, laplacian)
        return self.output_fc(self.attention(h))


class MBCGCNPredictor(nn.Module):
    """MBC-GCN: learnable time embedding concatenated with the four flow channels."""

    def __init__(self, time_feat_dim, k, gcn_hidden_dim, num_layers, dropout=0.1,
                 in_flow_dim=4, time_raw_dim=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_raw_dim, time_feat_dim),
            nn.GELU(),
            nn.Linear(time_feat_dim, time_feat_dim),
        )
        self.input_proj = nn.Linear(in_flow_dim + time_feat_dim, gcn_hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualSTGCNBlock(gcn_hidden_dim, gcn_hidden_dim, k, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attention = TemporalAttentionPool(gcn_hidden_dim)
        self.output_fc = nn.Sequential(
            nn.Linear(gcn_hidden_dim, gcn_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden_dim // 2, 2),
        )

    def forward(self, x, laplacian):
        flows, time_raw = x[..., :4], x[..., 4:]
        h = self.input_proj(torch.cat([flows, self.time_embed(time_raw)], dim=-1))
        for block in self.blocks:
            h = block(h, laplacian)
        return self.output_fc(self.attention(h))


def train_model(model, train_loader, val_loader, device, max_epochs, lr, patience,
                is_lstm=False, adjacency=None, weight_decay=0.0):
    """Train with early stopping on validation MSE; restores the best weights."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    model.to(device)

    for _ in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x) if is_lstm else model(x, adjacency)
            loss = criterion(pred.view_as(y), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x) if is_lstm else model(x, adjacency)
                val_loss += criterion(pred.view_as(y), y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss + 1e-6 < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_val


def build_5min_loaders(tensor, seq_len, batch_size):
    """Chronological 80/20 split; the validation window keeps its warm-up history."""
    split = int(tensor.shape[0] * 0.8)
    train_tensor, val_tensor = tensor[:split], tensor[split - seq_len :]

    return (
        DataLoader(FlattenedSequenceDataset(train_tensor, seq_len), batch_size, shuffle=True),
        DataLoader(FlattenedSequenceDataset(val_tensor, seq_len), batch_size, shuffle=False),
        DataLoader(GraphSequenceDataset(train_tensor, seq_len), batch_size, shuffle=True),
        DataLoader(GraphSequenceDataset(val_tensor, seq_len), batch_size, shuffle=False),
    )


def make_objective(model_name, tensor, adjacency, cfg, device, num_nodes, num_feats):
    """Optuna objective returning validation MSE for one architecture."""

    def objective(trial):
        seq_len = trial.suggest_categorical("seq_len", [6, 8, 12, 16, 24])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("wd", 1e-8, 1e-2, log=True)

        lstm_train, lstm_val, graph_train, graph_val = build_5min_loaders(
            tensor, seq_len, batch_size
        )

        if model_name == "LSTM":
            model = MultiNodeLSTM(
                in_dim=num_feats * num_nodes,
                hidden_dim=trial.suggest_categorical(
                    "hidden_dim", [64, 96, 128, 160, 192, 256, 320]
                ),
                num_layers=trial.suggest_int("num_layers", 1, 4),
                dropout=trial.suggest_float("dropout", 0.0, 0.5),
                num_nodes=num_nodes,
            )
            train_loader, val_loader = lstm_train, lstm_val
        elif model_name == "ST-GCN":
            model = STGCNPredictor(
                in_feat_dim=num_feats,
                hidden_dim=trial.suggest_categorical(
                    "hidden_dim", [32, 64, 80, 96, 128, 160, 192]
                ),
                num_layers=trial.suggest_int("num_layers", 1, 4),
                k=trial.suggest_int("K", 2, 6),
                dropout=trial.suggest_float("dropout", 0.0, 0.3),
            )
            train_loader, val_loader = graph_train, graph_val
        else:  # MBC-GCN
            model = MBCGCNPredictor(
                time_feat_dim=trial.suggest_categorical(
                    "time_feat_dim", [8, 16, 32, 48, 64, 96]
                ),
                k=trial.suggest_int("K", 2, 6),
                gcn_hidden_dim=trial.suggest_categorical(
                    "gcn_hidden_dim", [32, 64, 80, 96, 128, 160, 192]
                ),
                num_layers=trial.suggest_int("num_layers", 1, 4),
                dropout=trial.suggest_float("dropout", 0.0, 0.3),
            )
            train_loader, val_loader = graph_train, graph_val

        return train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            max_epochs=cfg.MAX_EPOCHS,
            lr=lr,
            patience=cfg.PATIENCE,
            is_lstm=(model_name == "LSTM"),
            adjacency=None if model_name == "LSTM" else adjacency.to(device),
            weight_decay=weight_decay,
        )

    return objective


def build_5min_model(model_name, params, num_nodes, num_feats):
    if model_name == "LSTM":
        return MultiNodeLSTM(
            in_dim=num_feats * num_nodes,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            num_nodes=num_nodes,
        )
    if model_name == "ST-GCN":
        return STGCNPredictor(
            in_feat_dim=num_feats,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            k=params["K"],
            dropout=params.get("dropout", 0.1),
        )
    return MBCGCNPredictor(
        time_feat_dim=params["time_feat_dim"],
        k=params["K"],
        gcn_hidden_dim=params["gcn_hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params.get("dropout", 0.1),
    )


def collect_5min_predictions(model, tensor, seq_len, batch_size, num_nodes, device,
                             is_lstm, adjacency=None):
    """Run the trained model over the validation window in scaled space."""
    split = int(tensor.shape[0] * 0.8)
    val_tensor = tensor[split - seq_len :]
    dataset = (
        FlattenedSequenceDataset(val_tensor, seq_len)
        if is_lstm
        else GraphSequenceDataset(val_tensor, seq_len)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)) if is_lstm else model(x.to(device), adjacency.to(device))
            trues.append(y.numpy())
            preds.append(pred.cpu().numpy())

    y_true = np.concatenate(trues, axis=0).reshape(-1, num_nodes, 2)
    y_pred = np.concatenate(preds, axis=0).reshape(-1, num_nodes, 2)
    return y_true, y_pred


def inverse_scale_per_node(scaled: np.ndarray, target_scalers: dict) -> np.ndarray:
    """Undo the per-station standardisation of the two target channels."""
    restored = np.zeros_like(scaled)
    for node in range(scaled.shape[1]):
        restored[:, node, :] = target_scalers[node].inverse_transform(scaled[:, node, :])
    return restored


def plot_total_flow(y_true, predictions: dict, out_dir: Path, window: int = 800,
                    y_max: float | None = None, filename: str = "total_flow.png"):
    """Network-wide observed and predicted curves for ride-in and ride-out."""
    steps = min(window, y_true.shape[0])
    total_true = y_true[:steps].sum(axis=1)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    styles = {
        "LSTM": ("--", "#ff7f0e"),
        "ST-GCN": (":", "#2ca02c"),
        "MBC-GCN": ("-.", "#d62728"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for index, name in enumerate(["Ride-in (Inflow)", "Ride-out (Outflow)"]):
        axes[index].plot(total_true[:, index], label="Actual", linewidth=1.8, color="#1f77b4")
        for model_name, y_pred in predictions.items():
            linestyle, colour = styles.get(model_name, ("--", None))
            axes[index].plot(
                y_pred[:steps].sum(axis=1)[:, index],
                label=model_name,
                linestyle=linestyle,
                linewidth=1.5,
                color=colour,
            )
        axes[index].set_title(f"Total Flow - {name}", fontweight="bold")
        axes[index].set_xlabel("Time index")
        axes[index].set_ylabel("Flow")
        if y_max is not None:
            axes[index].set_ylim(0, y_max)
        axes[index].legend(loc="upper right")
        axes[index].grid(alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_dir / filename}")


def run_5min(args) -> None:
    cfg = FiveMinConfig()
    cfg.N_TRIALS = args.n_trials
    device = get_device()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    tensor, adjacency, target_scalers, station_info = load_5min_panel(args.panel_dir, cfg)
    num_steps, num_nodes, num_feats = tensor.shape
    print(f"Data tensor: T={num_steps}, N={num_nodes}, F={num_feats}")
    station_info.to_csv(out_dir / "station_info.csv", index=False)

    metrics, predictions, y_true_inv = {}, {}, None

    for model_name in ["LSTM", "ST-GCN", "MBC-GCN"]:
        print(f"\n--- {model_name}: hyperparameter search ({cfg.N_TRIALS} trials) ---")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            make_objective(model_name, tensor, adjacency, cfg, device, num_nodes, num_feats),
            n_trials=cfg.N_TRIALS,
            show_progress_bar=True,
        )
        best = study.best_params
        print(f"Best validation MSE: {study.best_value:.6f}")
        print(f"Best parameters: {best}")

        seq_len = best.get("seq_len", cfg.SEQ_LEN)
        batch_size = best["batch_size"]
        is_lstm = model_name == "LSTM"

        lstm_train, lstm_val, graph_train, graph_val = build_5min_loaders(
            tensor, seq_len, batch_size
        )
        train_loader, val_loader = (
            (lstm_train, lstm_val) if is_lstm else (graph_train, graph_val)
        )

        model = build_5min_model(model_name, best, num_nodes, num_feats)
        train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            max_epochs=cfg.MAX_EPOCHS,
            lr=best["lr"],
            patience=cfg.PATIENCE,
            is_lstm=is_lstm,
            adjacency=None if is_lstm else adjacency.to(device),
            weight_decay=best.get("wd", 0.0),
        )

        y_true, y_pred = collect_5min_predictions(
            model, tensor, seq_len, cfg.BATCH_SIZE, num_nodes, device, is_lstm, adjacency
        )
        y_true_inv = inverse_scale_per_node(y_true, target_scalers)
        predictions[model_name] = inverse_scale_per_node(y_pred, target_scalers)
        metrics[model_name] = {
            "metrics": evaluate_flows(y_true_inv, predictions[model_name]),
            "best_params": best,
        }
        print(f"{model_name}: {metrics[model_name]['metrics']}")

        torch.save(model.state_dict(), out_dir / f"model_{model_name}.pt")

    # Different models may end up with different sequence lengths, so the
    # comparison window is truncated to the shortest validation series.
    shortest = min(v.shape[0] for v in predictions.values())
    y_true_inv = y_true_inv[:shortest]
    predictions = {k: v[:shortest] for k, v in predictions.items()}

    with open(out_dir / "metrics_5min.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=float)
    plot_total_flow(y_true_inv, predictions, out_dir, window=800, y_max=2250,
                    filename="total_flow_5min.png")
    print(f"\nAll outputs written to {out_dir}")


# ==========================================================================
# Scenario B: 1-hour network-wide forecasting
# ==========================================================================
class HourlyConfig:
    STATION_COL = "station__stationnames"
    TIME_COL = "hour"
    LON_COL = "station__lon"
    LAT_COL = "station__lat"

    TARGET_COLS = ["ride_in", "ride_out"]
    METRO_COLS = ["FINT_IN", "FINT_OUT"]

    SEQ_LENGTH = 12
    GCN_OUTPUT_DIM = 80
    NUM_GCN_LAYERS = 3
    TIME_FEAT_DIM = 32
    LSTM_HIDDEN = 160
    CHEB_K = 3

    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    LOSS_W_IN = 0.5
    LOSS_W_OUT = 0.5

    # Gaussian kernel parameters in kilometres.
    THRESHOLD_KM = 2.0
    SIGMA = 1.0


HOURLY_RENAME = {
    "ride_in_1h": "ride_in",
    "ride_out_1h": "ride_out",
    "FINT_IN_1h": "FINT_IN",
    "FINT_OUT_1h": "FINT_OUT",
}


def load_hourly_panel(csv_path: Path, cfg: HourlyConfig, use_metro: bool):
    """Build the hourly (T, N, F) tensor, the graph, and one scaler per flow channel.

    `use_metro=False` reproduces the autoregressive setting, in which the model
    sees only bike flows and calendar features.
    """
    df = pd.read_csv(csv_path).rename(columns=HOURLY_RENAME)
    if cfg.TIME_COL in df.columns and "FDT_STARTTIME" not in df.columns:
        df = df.rename(columns={cfg.TIME_COL: "FDT_STARTTIME"})
    df["FDT_STARTTIME"] = pd.to_datetime(df["FDT_STARTTIME"])

    station_info = (
        df[[cfg.STATION_COL, cfg.LON_COL, cfg.LAT_COL]].dropna().drop_duplicates()
    )
    num_nodes = len(station_info)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(
        station_info[cfg.LON_COL].values, station_info[cfg.LAT_COL].values
    )
    coords = np.vstack([xs, ys]).T
    dist_km = cdist(coords, coords) / 1000.0
    adj = np.exp(-(dist_km**2) / (2 * cfg.SIGMA**2))
    adj[dist_km > cfg.THRESHOLD_KM] = 0.0
    adjacency = normalise_adjacency(adj)

    df = df.sort_values(["FDT_STARTTIME", cfg.STATION_COL])
    df = add_cyclical_time_features(df, "FDT_STARTTIME")
    df["node_id"] = df[cfg.STATION_COL].rank(method="dense").astype(int) - 1
    df["time_id"] = df["FDT_STARTTIME"].rank(method="dense").astype(int) - 1

    flow_cols = cfg.TARGET_COLS + (cfg.METRO_COLS if use_metro else [])
    feature_cols = flow_cols + TIME_FEATURES
    num_steps = int(df["time_id"].max()) + 1

    tensor = np.zeros((num_steps, num_nodes, len(feature_cols)), dtype=np.float32)
    subset = df[["time_id", "node_id"] + feature_cols].dropna()
    tensor[subset["time_id"].to_numpy(), subset["node_id"].to_numpy(), :] = subset[
        feature_cols
    ].to_numpy(dtype=np.float32)

    # Each flow channel is standardised separately so that metro counts, which
    # are an order of magnitude larger, do not dominate the bike channels.
    scalers = {}
    for index, name in enumerate(flow_cols):
        scaler = StandardScaler()
        tensor[:, :, index] = scaler.fit_transform(
            tensor[:, :, index].reshape(-1, 1)
        ).reshape(num_steps, num_nodes)
        scalers[name] = scaler

    print(f"Data tensor: T={num_steps}, N={num_nodes}, F={len(feature_cols)}")
    return tensor, adjacency, scalers, num_nodes, len(feature_cols), len(flow_cols)


class JointFlowDataset(Dataset):
    """One step ahead, both targets at once, for every station."""

    def __init__(self, tensor: np.ndarray, seq_len: int):
        self.data = tensor
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, idx: int):
        window = self.data[idx : idx + self.seq_len]
        target = self.data[idx + self.seq_len, :, :2]
        return torch.from_numpy(window).float(), torch.from_numpy(target).float()


class STGCNBlock(nn.Module):
    """Temporal convolution around a Chebyshev graph convolution."""

    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.tcn1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.tcn2 = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.gcn = ChebGraphConv(out_channels, out_channels, k)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, laplacian):
        h1 = torch.relu(self.tcn1(x.permute(0, 3, 2, 1))).permute(0, 3, 2, 1)
        h_graph = torch.relu(self.gcn(h1, laplacian))
        h2 = self.tcn2(h_graph.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.norm(h1 + h2)


class HourlyMBCGCN(nn.Module):
    """MBC-GCN with a learnable time embedding over the flow channels."""

    def __init__(self, cfg, adjacency, num_flow_feats):
        super().__init__()
        self.adjacency = adjacency
        self.num_flow_feats = num_flow_feats
        self.time_embed = nn.Sequential(nn.Linear(4, cfg.TIME_FEAT_DIM), nn.ReLU())

        in_dim = num_flow_feats + cfg.TIME_FEAT_DIM
        blocks = [STGCNBlock(in_dim, cfg.GCN_OUTPUT_DIM, cfg.CHEB_K)]
        blocks += [
            STGCNBlock(cfg.GCN_OUTPUT_DIM, cfg.GCN_OUTPUT_DIM, cfg.CHEB_K)
            for _ in range(cfg.NUM_GCN_LAYERS - 1)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(cfg.GCN_OUTPUT_DIM * cfg.SEQ_LENGTH, 2)

    def forward(self, x):
        flows = x[..., : self.num_flow_feats]
        time_raw = x[..., self.num_flow_feats :]
        h = torch.cat([flows, self.time_embed(time_raw)], dim=-1)
        for block in self.blocks:
            h = block(h, self.adjacency)
        batch, steps, nodes, channels = h.shape
        return self.fc(h.transpose(1, 2).reshape(batch, nodes, steps * channels))


class HourlySTGCN(nn.Module):
    """Spatial baseline without the separate time-embedding branch."""

    def __init__(self, cfg, adjacency, input_feats):
        super().__init__()
        self.adjacency = adjacency
        blocks = [STGCNBlock(input_feats, cfg.GCN_OUTPUT_DIM, cfg.CHEB_K)]
        blocks += [
            STGCNBlock(cfg.GCN_OUTPUT_DIM, cfg.GCN_OUTPUT_DIM, cfg.CHEB_K)
            for _ in range(cfg.NUM_GCN_LAYERS - 1)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(cfg.GCN_OUTPUT_DIM * cfg.SEQ_LENGTH, 2)

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h, self.adjacency)
        batch, steps, nodes, channels = h.shape
        return self.fc(h.transpose(1, 2).reshape(batch, nodes, steps * channels))


class HourlyLSTM(nn.Module):
    """Per-station LSTM sharing weights across the network."""

    def __init__(self, cfg, input_feats):
        super().__init__()
        self.lstm = nn.LSTM(
            input_feats, cfg.LSTM_HIDDEN, cfg.NUM_GCN_LAYERS, batch_first=True
        )
        self.fc = nn.Linear(cfg.LSTM_HIDDEN, 2)

    def forward(self, x):
        batch, steps, nodes, feats = x.shape
        out, _ = self.lstm(x.reshape(batch * nodes, steps, feats))
        return self.fc(out[:, -1]).reshape(batch, nodes, 2)


def train_hourly(model, train_loader, val_loader, cfg, device):
    """Multi-task training with an equally weighted ride-in / ride-out loss."""
    optimizer = torch.optim.Adam(
        model.parameters(), cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    best_loss, best_state = float("inf"), None
    model.to(device)

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = cfg.LOSS_W_IN * F.mse_loss(pred[..., 0], y[..., 0]) + \
                cfg.LOSS_W_OUT * F.mse_loss(pred[..., 1], y[..., 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += (
                    cfg.LOSS_W_IN * F.mse_loss(pred[..., 0], y[..., 0])
                    + cfg.LOSS_W_OUT * F.mse_loss(pred[..., 1], y[..., 1])
                ).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss, best_state = val_loss, deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


def run_1h(args) -> None:
    cfg = HourlyConfig()
    device = get_device()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    use_metro = args.features == "semi_ar"
    print(f"Device: {device} | setting: {'Semi-AR (bike + metro)' if use_metro else 'AR (bike only)'}")

    tensor, adjacency, scalers, num_nodes, input_feats, num_flow_feats = load_hourly_panel(
        args.hourly_csv, cfg, use_metro
    )
    adjacency = adjacency.to(device)

    split = int(0.8 * tensor.shape[0])
    train_loader = DataLoader(
        JointFlowDataset(tensor[:split], cfg.SEQ_LENGTH), cfg.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        JointFlowDataset(tensor[split - cfg.SEQ_LENGTH :], cfg.SEQ_LENGTH), cfg.BATCH_SIZE
    )

    models = {
        "MBC-GCN": HourlyMBCGCN(cfg, adjacency, num_flow_feats),
        "ST-GCN": HourlySTGCN(cfg, adjacency, input_feats),
        "LSTM": HourlyLSTM(cfg, input_feats),
    }

    results, predictions, y_true = {}, {}, None
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        train_hourly(model, train_loader, val_loader, cfg, device)

        model.eval()
        trues, preds = [], []
        with torch.no_grad():
            for x, y in val_loader:
                trues.append(y.numpy())
                preds.append(model(x.to(device)).cpu().numpy())
        y_true_scaled = np.concatenate(trues)
        y_pred_scaled = np.concatenate(preds)

        y_true = np.zeros_like(y_true_scaled)
        y_pred = np.zeros_like(y_pred_scaled)
        for index, target in enumerate(cfg.TARGET_COLS):
            shape = y_true_scaled.shape[:-1]
            y_true[..., index] = (
                scalers[target]
                .inverse_transform(y_true_scaled[..., index].reshape(-1, 1))
                .reshape(shape)
            )
            y_pred[..., index] = (
                scalers[target]
                .inverse_transform(y_pred_scaled[..., index].reshape(-1, 1))
                .reshape(shape)
            )
        y_pred = np.clip(y_pred, 0, None)

        predictions[name] = y_pred
        results[name] = evaluate_flows(y_true, y_pred)
        for target, values in results[name].items():
            print(
                f"  {target}: MAE={values['MAE']:.2f} "
                f"RMSE={values['RMSE']:.2f} R2={values['R2']:.4f}"
            )
        torch.save(model.state_dict(), out_dir / f"model_{name}_{args.features}.pt")

    with open(out_dir / f"metrics_1h_{args.features}.json", "w") as handle:
        json.dump(results, handle, indent=2, default=float)
    plot_total_flow(
        y_true, predictions, out_dir, window=80,
        filename=f"total_flow_1h_{args.features}.png"
    )
    print(f"\nAll outputs written to {out_dir}")


# ==========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=["5min", "1h"], required=True)
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("outputs/panel/100m"),
        help="Directory of merged_YYYYMMDD.csv files (5-minute scenario).",
    )
    parser.add_argument(
        "--hourly-csv",
        type=Path,
        default=Path("outputs/panel/200m/20days_data_1h.csv"),
        help="Hourly station-time panel (1-hour scenario).",
    )
    parser.add_argument(
        "--features",
        choices=["ar", "semi_ar"],
        default="semi_ar",
        help="ar = bike flows only; semi_ar = bike flows plus metro flows.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/forecast_global"))
    parser.add_argument("--n-trials", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args()
    if args.resolution == "5min":
        run_5min(args)
    else:
        run_1h(args)


if __name__ == "__main__":
    main()
