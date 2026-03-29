"""Tests for market microstructure features."""

import numpy as np
import pandas as pd
import pytest

from src.features.microstructure import (
    compute_microprice,
    compute_ofi,
    compute_trade_flow_features,
    compute_vpin,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trades():
    """Create a realistic sample trades DataFrame."""
    np.random.seed(42)
    n = 200
    base_price = 50000.0
    prices = base_price + np.cumsum(np.random.randn(n) * 10)
    quantities = np.abs(np.random.lognormal(mean=-2, sigma=1, size=n))
    sides = np.random.choice(["buy", "sell"], size=n)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="100ms")

    return pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "price": prices,
        "quantity": quantities,
        "side": sides,
    })


@pytest.fixture
def sample_depth():
    """Create a realistic sample depth DataFrame."""
    rows = []
    base_price = 50000.0
    for snap_id in range(5):
        for i in range(10):
            rows.append({
                "timestamp": f"2025-01-01T00:00:{snap_id:02d}",
                "price": base_price - (i + 1) * 0.5,
                "quantity": np.random.uniform(0.1, 5.0),
                "side": "bid",
                "snapshot_id": snap_id,
            })
            rows.append({
                "timestamp": f"2025-01-01T00:00:{snap_id:02d}",
                "price": base_price + (i + 1) * 0.5,
                "quantity": np.random.uniform(0.1, 5.0),
                "side": "ask",
                "snapshot_id": snap_id,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VPIN Tests
# ---------------------------------------------------------------------------

class TestVPIN:
    def test_returns_series(self, sample_trades):
        result = compute_vpin(sample_trades, bucket_size=1.0)
        assert isinstance(result, pd.Series)
        assert result.name == "vpin"

    def test_values_in_range(self, sample_trades):
        result = compute_vpin(sample_trades, bucket_size=1.0)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_empty_input(self):
        result = compute_vpin(pd.DataFrame())
        assert len(result) == 0

    def test_n_buckets_override(self, sample_trades):
        result = compute_vpin(sample_trades, n_buckets=10)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# OFI Tests
# ---------------------------------------------------------------------------

class TestOFI:
    def test_returns_dataframe(self, sample_depth):
        result = compute_ofi(sample_depth, n_levels=5)
        assert isinstance(result, pd.DataFrame)
        assert "ofi" in result.columns
        assert "bid_depth" in result.columns
        assert "ask_depth" in result.columns

    def test_ofi_range(self, sample_depth):
        result = compute_ofi(sample_depth, n_levels=5)
        assert (result["ofi"] >= -1).all()
        assert (result["ofi"] <= 1).all()

    def test_one_row_per_snapshot(self, sample_depth):
        result = compute_ofi(sample_depth, n_levels=5)
        assert len(result) == sample_depth["snapshot_id"].nunique()

    def test_empty_input(self):
        result = compute_ofi(pd.DataFrame())
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Microprice Tests
# ---------------------------------------------------------------------------

class TestMicroprice:
    def test_returns_dataframe(self, sample_depth):
        result = compute_microprice(sample_depth, n_levels=5)
        assert isinstance(result, pd.DataFrame)
        assert "microprice" in result.columns
        assert "midprice" in result.columns
        assert "spread_bps" in result.columns

    def test_microprice_between_bid_ask(self, sample_depth):
        result = compute_microprice(sample_depth, n_levels=5)
        # Microprice should be within the bid-ask range
        for _, row in result.iterrows():
            snap = sample_depth[sample_depth["snapshot_id"] == row["snapshot_id"]]
            best_bid = snap[snap["side"] == "bid"]["price"].max()
            best_ask = snap[snap["side"] == "ask"]["price"].min()
            assert best_bid <= row["microprice"] <= best_ask

    def test_spread_positive(self, sample_depth):
        result = compute_microprice(sample_depth, n_levels=5)
        assert (result["spread_bps"] >= 0).all()

    def test_empty_input(self):
        result = compute_microprice(pd.DataFrame())
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Trade Flow Features Tests
# ---------------------------------------------------------------------------

class TestTradeFlowFeatures:
    def test_returns_dataframe(self, sample_trades):
        result = compute_trade_flow_features(sample_trades, window=20)
        assert isinstance(result, pd.DataFrame)
        assert "inter_arrival" in result.columns
        assert "rolling_volatility" in result.columns
        assert "trade_intensity" in result.columns

    def test_same_length(self, sample_trades):
        result = compute_trade_flow_features(sample_trades, window=20)
        assert len(result) == len(sample_trades)

    def test_inter_arrival_non_negative(self, sample_trades):
        result = compute_trade_flow_features(sample_trades, window=20)
        assert (result["inter_arrival"] >= 0).all()
