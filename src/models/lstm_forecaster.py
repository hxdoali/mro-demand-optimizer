"""
PyTorch LSTM demand forecaster.

Provides a neural network alternative to Prophet for time series
forecasting, using a sequence-to-one LSTM architecture.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DemandDataset(Dataset):
    """Sliding window dataset for time series sequences."""

    def __init__(self, data: np.ndarray, seq_length: int = 28):
        self.data = data.astype(np.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)


class LSTMForecaster(nn.Module):
    """Single-layer LSTM for univariate demand forecasting."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def normalize(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalize to [0, 1]."""
    d_min, d_max = data.min(), data.max()
    if d_max - d_min == 0:
        return np.zeros_like(data), d_min, d_max
    return (data - d_min) / (d_max - d_min), d_min, d_max


def denormalize(data: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    return data * (d_max - d_min) + d_min


def train_lstm(
    demand_series: np.ndarray,
    seq_length: int = 28,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str | None = None,
) -> tuple[LSTMForecaster, float, float]:
    """
    Train an LSTM model on a single SKU's demand history.

    Returns the trained model and normalization parameters.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    normed, d_min, d_max = normalize(demand_series)

    dataset = DemandDataset(normed, seq_length)
    train_size = int(len(dataset) * 0.85)
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = LSTMForecaster(hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()

        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return model, d_min, d_max


def forecast_lstm(
    model: LSTMForecaster,
    recent_data: np.ndarray,
    horizon: int = 30,
    d_min: float = 0.0,
    d_max: float = 1.0,
    device: str | None = None,
) -> np.ndarray:
    """Generate multi-step forecast by autoregressively feeding predictions."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    seq_length = 28
    normed, _, _ = normalize(recent_data)
    window = normed[-seq_length:].copy()

    predictions = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x).item()
            predictions.append(pred)
            window = np.append(window[1:], pred)

    raw_preds = denormalize(np.array(predictions), d_min, d_max)
    return np.clip(raw_preds, 0, None).round().astype(int)


def forecast_sku_lstm(
    demand_df: pd.DataFrame,
    sku_id: str,
    horizon_days: int = 30,
    epochs: int = 50,
) -> dict:
    """
    End-to-end LSTM forecast for a single SKU.
    Comparable interface to forecaster.forecast_sku() for easy swapping.
    """
    sku_data = demand_df[demand_df["sku_id"] == sku_id].sort_values("date")
    demand_values = sku_data["demand"].values.astype(float)
    dates = pd.to_datetime(sku_data["date"].values)

    train_end_idx = len(demand_values) - horizon_days
    train_data = demand_values[:train_end_idx]
    test_data = demand_values[train_end_idx:]

    model, d_min, d_max = train_lstm(train_data, epochs=epochs)

    predictions = forecast_lstm(model, train_data, horizon=horizon_days, d_min=d_min, d_max=d_max)

    future_dates = pd.date_range(start=dates[train_end_idx], periods=horizon_days, freq="D")

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_demand": predictions,
        "sku_id": sku_id,
    })

    return {
        "sku_id": sku_id,
        "model": model,
        "forecast": forecast_df,
        "train_size": len(train_data),
        "test_actual": test_data[:horizon_days] if len(test_data) >= horizon_days else test_data,
        "horizon_days": horizon_days,
    }
