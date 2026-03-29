"""
Causal Inference Layer for market impact estimation.

Implements Double Machine Learning (DML) to isolate the causal effect
of dark pool executions on price, and adversarial debiasing to correct
for selection bias in observed fill data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Double Machine Learning for Market Impact
# ---------------------------------------------------------------------------

@dataclass
class CausalEstimate:
    """Container for causal estimation results."""
    ate: float                            # Average Treatment Effect
    ate_std: float                        # Standard error of ATE
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    treatment_col: str = ""
    outcome_col: str = ""
    n_samples: int = 0
    nuisance_r2_treatment: float = 0.0    # R² of treatment nuisance model
    nuisance_r2_outcome: float = 0.0      # R² of outcome nuisance model


class MarketImpactEstimator:
    """
    Double Machine Learning estimator for causal market impact.

    Isolates the treatment effect of a dark pool execution (binary treatment)
    on price impact (continuous outcome), controlling for confounders like
    global volatility, order flow, and venue state.

    Implements the Partially Linear Model (PLM) variant of DML:
        Y = θ·T + g(X) + ε
        T = m(X) + η

    Where:
        Y = price impact (bps)
        T = dark pool fill indicator
        X = confounders (volatility, OFI, spread, etc.)
        θ = causal effect (what we want)
    """

    def __init__(
        self,
        n_folds: int = 3,
        nuisance_model: str = "gradient_boosting",
    ):
        self.n_folds = n_folds
        self.nuisance_model = nuisance_model

    def _get_nuisance_model(self, task: str = "regression"):
        """Factory for nuisance models."""
        if self.nuisance_model == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
        elif self.nuisance_model == "random_forest":
            return RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42,
            )
        else:
            raise ValueError(f"Unknown nuisance model: {self.nuisance_model}")

    def estimate(
        self,
        data: pd.DataFrame,
        treatment_col: str = "is_dark_fill",
        outcome_col: str = "price_impact_bps",
        confounder_cols: list[str] | None = None,
    ) -> CausalEstimate:
        """
        Estimate the Average Treatment Effect using cross-fitted DML.

        Args:
            data: DataFrame containing treatment, outcome, and confounders.
            treatment_col: Binary treatment column name.
            outcome_col: Continuous outcome column name.
            confounder_cols: List of confounder column names. If None,
                            uses all numeric columns except treatment/outcome.

        Returns:
            CausalEstimate with ATE and diagnostics.
        """
        df = data.dropna(subset=[treatment_col, outcome_col]).copy()

        if confounder_cols is None:
            exclude = {treatment_col, outcome_col}
            confounder_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                               if c not in exclude]

        X = df[confounder_cols].values.astype(np.float64)
        T = df[treatment_col].values.astype(np.float64)
        Y = df[outcome_col].values.astype(np.float64)

        n = len(df)
        if n < self.n_folds * 2:
            logger.warning(f"Only {n} samples for DML; results may be unreliable")

        # Cross-fitted residualization
        residuals_Y = np.zeros(n)
        residuals_T = np.zeros(n)
        r2_outcome_scores = []
        r2_treatment_scores = []

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X):
            # Fit nuisance models on training fold
            model_Y = self._get_nuisance_model()
            model_T = self._get_nuisance_model()

            model_Y.fit(X[train_idx], Y[train_idx])
            model_T.fit(X[train_idx], T[train_idx])

            # Predict on test fold
            Y_hat = model_Y.predict(X[test_idx])
            T_hat = model_T.predict(X[test_idx])

            residuals_Y[test_idx] = Y[test_idx] - Y_hat
            residuals_T[test_idx] = T[test_idx] - T_hat

            # Track R² for diagnostics
            r2_outcome_scores.append(model_Y.score(X[test_idx], Y[test_idx]))
            r2_treatment_scores.append(model_T.score(X[test_idx], T[test_idx]))

        # Final stage: regress Y-residuals on T-residuals
        # θ = Σ(ε_T · ε_Y) / Σ(ε_T²)
        denom = (residuals_T ** 2).sum()
        if denom < 1e-10:
            logger.warning("Treatment residuals near zero — all confounders explain treatment")
            ate = 0.0
            ate_std = float("inf")
        else:
            ate = (residuals_T * residuals_Y).sum() / denom
            # Standard error via heteroskedasticity-robust formula
            se_num = ((residuals_T ** 2) * (residuals_Y - ate * residuals_T) ** 2).sum()
            ate_std = np.sqrt(se_num) / denom

        ci_low = ate - 1.96 * ate_std
        ci_high = ate + 1.96 * ate_std

        logger.info(f"DML ATE: {ate:.4f} ± {ate_std:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        return CausalEstimate(
            ate=float(ate),
            ate_std=float(ate_std),
            confidence_interval=(float(ci_low), float(ci_high)),
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            n_samples=n,
            nuisance_r2_treatment=float(np.mean(r2_treatment_scores)),
            nuisance_r2_outcome=float(np.mean(r2_outcome_scores)),
        )


# ---------------------------------------------------------------------------
# Adversarial Debiasing
# ---------------------------------------------------------------------------

class AdversarialDebias(nn.Module):
    """
    Adversarial debiasing network to detect and correct selection bias.

    In dark pool data, we only observe fills (selection on outcome).
    This discriminator learns to predict whether an observation was
    "selected" (observed as a fill) vs. "hidden" (liquidity present
    but no fill). The learned features are then used to reweight the
    training data.

    Architecture: MLP discriminator trained adversarially — the main
    model tries to make its representations uninformative about
    selection status, while the discriminator tries to predict it.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict selection probability (is this a visible fill?).

        Args:
            features: (N, feature_dim) feature vectors.

        Returns:
            logits: (N,) raw selection logits.
        """
        return self.discriminator(features).squeeze(-1)

    def compute_selection_weights(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute inverse propensity weights to correct selection bias.

        For observed fills, weight = 1/P(selected|X).
        Clips weights to avoid extreme values.
        """
        self.eval()
        with torch.no_grad():
            logits = self(features)
            p_selected = torch.sigmoid(logits).clamp(0.05, 0.95)
            weights = 1.0 / p_selected
            # Normalize to have mean 1
            weights = weights / weights.mean()
        return weights

    def train_discriminator(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 20,
        lr: float = 0.0005,
        batch_size: int = 64,
    ) -> list[float]:
        """
        Train the selection discriminator.

        Args:
            features: (N, D) feature matrix.
            labels: (N,) binary labels (1=observed fill, 0=no fill).
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.

        Returns:
            List of per-epoch losses.
        """
        self.train()
        X = torch.from_numpy(features.astype(np.float32))
        y = torch.from_numpy(labels.astype(np.float32))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses = []
        n = len(X)

        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                batch_x, batch_y = X[idx], y[idx]

                logits = self(batch_x)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                logger.info(f"  AdversarialDebias epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        return losses


# ---------------------------------------------------------------------------
# Combined fill probability estimator
# ---------------------------------------------------------------------------

def estimate_fill_probability(
    tpp_fill_prob: np.ndarray,
    causal_ate: float,
    selection_weights: np.ndarray | None = None,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Combine TPP intensity-based fill probability with causal ATE adjustment.

    The final fill probability accounts for:
    1. TPP model prediction (how likely is a fill based on event dynamics)
    2. Causal market impact (how much does our order move the price)
    3. Selection bias correction (inverse propensity weighting)

    Args:
        tpp_fill_prob: (N,) array of fill probabilities from TPP model.
        causal_ate: Average treatment effect estimate from DML.
        selection_weights: (N,) optional IPW weights from adversarial debias.
        alpha: Weight for TPP vs. causal adjustment (0=all causal, 1=all TPP).

    Returns:
        adjusted_prob: (N,) adjusted fill probabilities in [0, 1].
    """
    # Causal adjustment: if ATE is negative (our trade reduces price impact),
    # that's favorable — increase fill probability. If positive, decrease.
    causal_adjustment = np.clip(-causal_ate / 10.0, -0.2, 0.2)  # Scale to ±20%

    adjusted = alpha * tpp_fill_prob + (1 - alpha) * (tpp_fill_prob + causal_adjustment)

    # Apply selection bias correction
    if selection_weights is not None:
        # Higher weight = more likely to be a "hidden" fill → boost probability
        weight_adjustment = (selection_weights - 1.0) * 0.05
        adjusted = adjusted + weight_adjustment

    return np.clip(adjusted, 0.0, 1.0)
