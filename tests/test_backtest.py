"""Tests for the backtest engine."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.backtest.engine import (
    BacktestMetrics,
    DarkPoolSimulator,
    run_backtest,
    slippage_linear,
    slippage_sqrt,
)
from src.model.tpp_core import NeuralTPP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trades():
    """Create trades with enough events for a backtest."""
    np.random.seed(42)
    n = 500
    prices = 50000 + np.cumsum(np.random.randn(n) * 10)
    quantities = np.abs(np.random.lognormal(mean=-2, sigma=1, size=n))
    sides = np.random.choice(["buy", "sell"], size=n)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1s")

    return pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "price": prices,
        "quantity": quantities,
        "side": sides,
    })


@pytest.fixture
def model():
    return NeuralTPP(feature_dim=2, hidden_dim=16, num_layers=1)


@pytest.fixture
def config():
    return {
        "backtest": {
            "initial_capital_usd": 100_000,
            "block_size_btc": 0.1,
            "fill_threshold": 0.3,
            "warmup_events": 50,
            "slippage_model": "sqrt",
            "slippage_base_bps": 0.5,
        },
        "training": {
            "sequence_length": 64,
        },
    }


# ---------------------------------------------------------------------------
# Slippage Tests
# ---------------------------------------------------------------------------

class TestSlippage:
    def test_linear_slippage(self):
        s = slippage_linear(1.0, base_bps=0.5)
        assert s == 0.5

    def test_sqrt_slippage(self):
        s = slippage_sqrt(4.0, base_bps=1.0)
        assert abs(s - 2.0) < 1e-6

    def test_slippage_increases_with_quantity(self):
        s1 = slippage_sqrt(1.0)
        s2 = slippage_sqrt(10.0)
        assert s2 > s1


# ---------------------------------------------------------------------------
# DarkPoolSimulator Tests
# ---------------------------------------------------------------------------

class TestDarkPoolSimulator:
    def test_runs_without_error(self, model, sample_trades, config):
        metrics = run_backtest(model, sample_trades, config)
        assert isinstance(metrics, BacktestMetrics)

    def test_metrics_nonnegative(self, model, sample_trades, config):
        metrics = run_backtest(model, sample_trades, config)
        assert metrics.total_orders >= 0
        assert metrics.filled_orders >= 0
        assert metrics.fill_rate >= 0
        assert metrics.total_volume_btc >= 0

    def test_fill_rate_bounded(self, model, sample_trades, config):
        metrics = run_backtest(model, sample_trades, config)
        assert 0 <= metrics.fill_rate <= 1

    def test_summary_string(self, model, sample_trades, config):
        metrics = run_backtest(model, sample_trades, config)
        summary = metrics.summary()
        assert isinstance(summary, str)
        assert "Backtest Summary" in summary

    def test_too_few_events(self, model, config):
        trades = pd.DataFrame({
            "timestamp": ["2025-01-01"] * 5,
            "price": [50000] * 5,
            "quantity": [0.1] * 5,
            "side": ["buy"] * 5,
        })
        metrics = run_backtest(model, trades, config)
        assert metrics.total_orders == 0
