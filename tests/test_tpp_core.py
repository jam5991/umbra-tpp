"""Tests for the Neural TPP model."""

import numpy as np
import torch
import pytest

from src.model.tpp_core import (
    IntensityRNN,
    MarkPredictor,
    NeuralTPP,
    TPPBatch,
    TPPDataset,
    collate_tpp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch():
    """Create a sample TPPBatch."""
    B, L, F = 4, 32, 2
    return TPPBatch(
        inter_arrival_times=torch.rand(B, L),
        marks=torch.rand(B, L) * 0.1,
        features=torch.randn(B, L, F),
        mask=torch.ones(B, L),
    )


@pytest.fixture
def model():
    """Create a NeuralTPP model."""
    return NeuralTPP(feature_dim=2, hidden_dim=32, num_layers=1, mark_dim=1)


# ---------------------------------------------------------------------------
# IntensityRNN Tests
# ---------------------------------------------------------------------------

class TestIntensityRNN:
    def test_output_shape(self):
        rnn = IntensityRNN(input_dim=2, hidden_dim=32, num_layers=1)
        B, L = 4, 32
        ia = torch.rand(B, L)
        marks = torch.rand(B, L)
        features = torch.randn(B, L, 2)

        intensities, hidden = rnn(ia, marks, features)
        assert intensities.shape == (B, L)
        assert hidden.shape == (B, L, 32)

    def test_positive_intensities(self):
        rnn = IntensityRNN(input_dim=2, hidden_dim=32)
        B, L = 4, 32
        ia = torch.rand(B, L)
        marks = torch.rand(B, L)
        features = torch.randn(B, L, 2)

        intensities, _ = rnn(ia, marks, features)
        assert (intensities > 0).all(), "Intensities must be positive"

    def test_with_mask(self):
        rnn = IntensityRNN(input_dim=2, hidden_dim=32)
        B, L = 4, 32
        ia = torch.rand(B, L)
        marks = torch.rand(B, L)
        features = torch.randn(B, L, 2)
        mask = torch.ones(B, L)
        mask[:, 20:] = 0  # Variable length

        intensities, hidden = rnn(ia, marks, features, mask)
        assert intensities.shape == (B, L)


# ---------------------------------------------------------------------------
# MarkPredictor Tests
# ---------------------------------------------------------------------------

class TestMarkPredictor:
    def test_output_shape(self):
        mp = MarkPredictor(hidden_dim=32, mark_dim=1)
        hidden = torch.randn(4, 32, 32)
        mu, sigma = mp(hidden)
        assert mu.shape == (4, 32, 1)
        assert sigma.shape == (4, 32, 1)

    def test_sigma_positive(self):
        mp = MarkPredictor(hidden_dim=32, mark_dim=1)
        hidden = torch.randn(4, 32, 32)
        _, sigma = mp(hidden)
        assert (sigma > 0).all(), "Sigma must be positive"

    def test_log_prob(self):
        mp = MarkPredictor(hidden_dim=32, mark_dim=1)
        hidden = torch.randn(4, 32, 32)
        marks = torch.rand(4, 32) * 0.1 + 0.01  # Positive marks
        log_prob = mp.log_prob(hidden, marks)
        assert log_prob.shape == (4, 32)
        assert torch.isfinite(log_prob).all()


# ---------------------------------------------------------------------------
# NeuralTPP Tests
# ---------------------------------------------------------------------------

class TestNeuralTPP:
    def test_forward_keys(self, model, batch):
        output = model(batch)
        expected_keys = {"intensities", "mark_mu", "mark_sigma",
                         "temporal_nll", "mark_nll", "loss"}
        assert set(output.keys()) == expected_keys

    def test_loss_finite(self, model, batch):
        output = model(batch)
        assert torch.isfinite(output["loss"])
        assert torch.isfinite(output["temporal_nll"])
        assert torch.isfinite(output["mark_nll"])

    def test_gradient_flow(self, model, batch):
        output = model(batch)
        output["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"

    def test_fill_probability_range(self, model, batch):
        probs = model.predict_fill_probability(batch, horizon=1.0)
        assert probs.shape == (batch.inter_arrival_times.shape[0],)
        assert (probs >= 0).all()
        assert (probs <= 1).all()


# ---------------------------------------------------------------------------
# Dataset Tests
# ---------------------------------------------------------------------------

class TestTPPDataset:
    def test_dataset_creation(self):
        import pandas as pd
        n = 500
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="100ms").astype(str),
            "price": 50000 + np.cumsum(np.random.randn(n)),
            "quantity": np.abs(np.random.lognormal(-2, 1, n)),
            "side": np.random.choice(["buy", "sell"], n),
        })

        ds = TPPDataset(df, seq_length=64)
        assert len(ds) > 0

        item = ds[0]
        assert "inter_arrival_times" in item
        assert "marks" in item
        assert "features" in item
        assert "mask" in item
        assert item["inter_arrival_times"].shape == (64,)

    def test_collate(self):
        import pandas as pd
        n = 200
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="100ms").astype(str),
            "price": 50000 + np.cumsum(np.random.randn(n)),
            "quantity": np.abs(np.random.lognormal(-2, 1, n)),
            "side": np.random.choice(["buy", "sell"], n),
        })

        ds = TPPDataset(df, seq_length=32)
        batch = collate_tpp([ds[i] for i in range(min(4, len(ds)))])
        assert isinstance(batch, TPPBatch)
        assert batch.inter_arrival_times.ndim == 2
