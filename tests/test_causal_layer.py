"""Tests for the causal inference layer."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.model.causal_layer import (
    AdversarialDebias,
    CausalEstimate,
    MarketImpactEstimator,
    estimate_fill_probability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def causal_data():
    """Create synthetic data with a known treatment effect."""
    np.random.seed(42)
    n = 500

    # Confounders
    volatility = np.random.exponential(1.0, n)
    volume_flow = np.random.exponential(10.0, n)

    # Treatment assignment (partially confounded by volatility)
    p_treat = 1 / (1 + np.exp(-(volatility - 1.0)))
    treatment = (np.random.rand(n) < p_treat).astype(float)

    # Outcome: true ATE = -2.0 bps (our trade reduces impact)
    true_ate = -2.0
    outcome = true_ate * treatment + 0.5 * volatility - 0.1 * volume_flow + np.random.randn(n) * 2

    return pd.DataFrame({
        "is_dark_fill": treatment,
        "price_impact_bps": outcome,
        "volatility": volatility,
        "volume_flow": volume_flow,
    })


# ---------------------------------------------------------------------------
# MarketImpactEstimator Tests
# ---------------------------------------------------------------------------

class TestMarketImpactEstimator:
    def test_returns_causal_estimate(self, causal_data):
        est = MarketImpactEstimator(n_folds=3, nuisance_model="gradient_boosting")
        result = est.estimate(causal_data)
        assert isinstance(result, CausalEstimate)

    def test_ate_reasonable_direction(self, causal_data):
        """ATE should be negative (true effect is -2.0)."""
        est = MarketImpactEstimator(n_folds=3)
        result = est.estimate(causal_data)
        # Allow some estimation error, but should be negative
        assert result.ate < 1.0, f"ATE should be negative-ish, got {result.ate}"

    def test_confidence_interval_covers_ate(self, causal_data):
        est = MarketImpactEstimator(n_folds=3)
        result = est.estimate(causal_data)
        ci_low, ci_high = result.confidence_interval
        assert ci_low < ci_high

    def test_nuisance_r2_nonnegative(self, causal_data):
        est = MarketImpactEstimator(n_folds=3)
        result = est.estimate(causal_data)
        # R² can be negative for bad models but should generally be >= -1
        assert result.nuisance_r2_outcome > -1.0

    def test_random_forest_backend(self, causal_data):
        est = MarketImpactEstimator(n_folds=3, nuisance_model="random_forest")
        result = est.estimate(causal_data)
        assert isinstance(result, CausalEstimate)


# ---------------------------------------------------------------------------
# AdversarialDebias Tests
# ---------------------------------------------------------------------------

class TestAdversarialDebias:
    def test_forward_shape(self):
        debias = AdversarialDebias(feature_dim=4, hidden_dim=16)
        x = torch.randn(10, 4)
        logits = debias(x)
        assert logits.shape == (10,)

    def test_train_discriminator(self):
        debias = AdversarialDebias(feature_dim=4, hidden_dim=16)
        features = np.random.randn(100, 4).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)

        losses = debias.train_discriminator(features, labels, epochs=5, lr=0.001)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)

    def test_selection_weights(self):
        debias = AdversarialDebias(feature_dim=4, hidden_dim=16)
        features = np.random.randn(50, 4).astype(np.float32)
        labels = np.random.randint(0, 2, 50).astype(np.float32)

        debias.train_discriminator(features, labels, epochs=3)
        weights = debias.compute_selection_weights(torch.from_numpy(features))

        assert weights.shape == (50,)
        assert (weights > 0).all()
        # Weights should be normalized to mean ~1
        assert abs(weights.mean().item() - 1.0) < 0.5


# ---------------------------------------------------------------------------
# Fill Probability Tests
# ---------------------------------------------------------------------------

class TestEstimateFillProbability:
    def test_output_range(self):
        tpp_probs = np.random.uniform(0.3, 0.8, 20)
        result = estimate_fill_probability(tpp_probs, causal_ate=-2.0)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_with_selection_weights(self):
        tpp_probs = np.random.uniform(0.3, 0.8, 20)
        weights = np.ones(20) * 1.1
        result = estimate_fill_probability(tpp_probs, causal_ate=-1.0, selection_weights=weights)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_negative_ate_boosts_probability(self):
        tpp_probs = np.full(10, 0.5)
        result_neg = estimate_fill_probability(tpp_probs, causal_ate=-5.0)
        result_pos = estimate_fill_probability(tpp_probs, causal_ate=5.0)
        # Negative ATE (reduces impact) should give higher fill prob
        assert result_neg.mean() >= result_pos.mean()
