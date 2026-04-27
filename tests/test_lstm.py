"""Tests for PyTorch LSTM forecaster."""

import pytest
import numpy as np
from src.models.lstm_forecaster import (
    DemandDataset,
    LSTMForecaster,
    normalize,
    denormalize,
    train_lstm,
    forecast_lstm,
)


class TestNormalization:
    def test_normalize_roundtrip(self):
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        normed, d_min, d_max = normalize(data)
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)
        restored = denormalize(normed, d_min, d_max)
        np.testing.assert_array_almost_equal(restored, data)

    def test_normalize_constant(self):
        data = np.array([5, 5, 5], dtype=float)
        normed, _, _ = normalize(data)
        assert (normed == 0).all()


class TestDemandDataset:
    def test_dataset_length(self):
        data = np.arange(100, dtype=float)
        ds = DemandDataset(data, seq_length=10)
        assert len(ds) == 90

    def test_dataset_shapes(self):
        data = np.arange(50, dtype=float)
        ds = DemandDataset(data, seq_length=7)
        x, y = ds[0]
        assert x.shape == (7, 1)
        assert y.shape == ()


class TestLSTMModel:
    def test_forward_pass(self):
        model = LSTMForecaster(input_size=1, hidden_size=16, num_layers=1)
        x = np.random.rand(4, 28, 1).astype(np.float32)
        import torch
        out = model(torch.tensor(x))
        assert out.shape == (4,)

    def test_train_and_forecast(self):
        np.random.seed(42)
        data = np.sin(np.linspace(0, 8 * np.pi, 200)) * 50 + 60
        data = data + np.random.normal(0, 5, 200)

        model, d_min, d_max = train_lstm(data, seq_length=14, hidden_size=16, num_layers=1, epochs=5, batch_size=16)
        preds = forecast_lstm(model, data, horizon=7, d_min=d_min, d_max=d_max)

        assert len(preds) == 7
        assert all(p >= 0 for p in preds)
