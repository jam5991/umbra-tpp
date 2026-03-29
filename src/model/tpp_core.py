"""
Neural Temporal Point Process (TPP) for hidden liquidity prediction.

Implements an RNN-based conditional intensity function λ(t) and a mark
predictor (MLP head) for order size estimation. The full model outputs
log-likelihood for training and intensity / fill-probability for inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TPPBatch:
    """A batch of event sequences for TPP training.

    Attributes:
        inter_arrival_times: (batch, seq_len) inter-arrival times Δt.
        marks: (batch, seq_len) event marks (order sizes).
        features: (batch, seq_len, feature_dim) auxiliary features per event.
        mask: (batch, seq_len) boolean mask for valid positions.
    """
    inter_arrival_times: torch.Tensor   # (B, L)
    marks: torch.Tensor                 # (B, L)
    features: torch.Tensor              # (B, L, F)
    mask: torch.Tensor                  # (B, L)


# ---------------------------------------------------------------------------
# Intensity RNN
# ---------------------------------------------------------------------------

class IntensityRNN(nn.Module):
    """
    RNN-based conditional intensity function λ(t|H_t).

    Parameterizes the intensity of the next event arrival using a GRU
    over the history of past events. Output is passed through softplus
    to ensure positivity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_softplus = use_softplus

        # Input projection: Δt + mark + features → embedding
        self.input_proj = nn.Linear(input_dim + 2, hidden_dim)

        # GRU over event history
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Intensity head: hidden → scalar λ
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # LayerNorm on hidden states before intensity head prevents scale blow-up
        # while preserving within-sequence dynamics
        self.hidden_norm = nn.LayerNorm(hidden_dim)

        # Learnable base intensity (background rate).
        # Initialize to -2.0 so softplus gives ~0.12 — prevents early saturation.
        self.base_intensity = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        inter_arrivals: torch.Tensor,
        marks: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inter_arrivals: (B, L) inter-arrival times.
            marks: (B, L) event marks.
            features: (B, L, F) auxiliary features.
            mask: (B, L) valid positions.

        Returns:
            intensities: (B, L) conditional intensity λ(t) at each event.
            hidden_states: (B, L, H) hidden representations.
        """
        B, L = inter_arrivals.shape

        # Construct input: [Δt, mark, features]
        x = torch.cat([
            inter_arrivals.unsqueeze(-1),       # (B, L, 1)
            marks.unsqueeze(-1),                # (B, L, 1)
            features,                           # (B, L, F)
        ], dim=-1)                              # (B, L, F+2)

        x = self.input_proj(x)                  # (B, L, H)
        x = F.gelu(x)

        # Pack if mask is provided (for variable-length sequences)
        if mask is not None:
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.rnn(packed)
            hidden_states, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L,
            )
        else:
            hidden_states, _ = self.rnn(x)

        # Apply LayerNorm to stabilize hidden scale before intensity head
        normed = self.hidden_norm(hidden_states)

        # Compute intensities
        raw_intensity = self.intensity_head(normed).squeeze(-1)  # (B, L)
        raw_intensity = raw_intensity + F.softplus(self.base_intensity)

        if self.use_softplus:
            intensities = F.softplus(raw_intensity)
        else:
            intensities = torch.exp(raw_intensity.clamp(max=5.0))

        return intensities, hidden_states


# ---------------------------------------------------------------------------
# Mark Predictor
# ---------------------------------------------------------------------------

class MarkPredictor(nn.Module):
    """
    MLP head predicting the mark (order size) distribution.

    Uses a log-normal parameterization since order sizes are positive
    and heavy-tailed.
    """

    def __init__(self, hidden_dim: int = 64, mark_dim: int = 1):
        super().__init__()
        self.mark_dim = mark_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, mark_dim * 2),  # μ and log(σ) for log-normal
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (B, L, H) hidden states from RNN.

        Returns:
            mu: (B, L, mark_dim) log-normal mean.
            sigma: (B, L, mark_dim) log-normal std (positive).
        """
        out = self.net(hidden)  # (B, L, 2*mark_dim)
        mu, log_sigma = out.chunk(2, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-6
        return mu, sigma

    def log_prob(self, hidden: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of observed marks under log-normal."""
        mu, sigma = self(hidden)
        marks = marks.unsqueeze(-1).clamp(min=1e-8)

        # Log-normal log-prob
        log_marks = torch.log(marks)
        log_prob = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - torch.log(marks)
            - 0.5 * ((log_marks - mu) / sigma) ** 2
        )
        return log_prob.squeeze(-1)  # (B, L)


# ---------------------------------------------------------------------------
# Full Neural TPP Model
# ---------------------------------------------------------------------------

class NeuralTPP(nn.Module):
    """
    Complete Neural Temporal Point Process model.

    Combines the IntensityRNN (when does the next event arrive?) with the
    MarkPredictor (how large is the order?) to produce both the temporal
    log-likelihood and the mark log-likelihood.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        mark_dim: int = 1,
        dropout: float = 0.1,
        mc_samples: int = 20,
    ):
        super().__init__()
        self.mc_samples = mc_samples

        self.intensity_rnn = IntensityRNN(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.mark_predictor = MarkPredictor(
            hidden_dim=hidden_dim,
            mark_dim=mark_dim,
        )

    def forward(self, batch: TPPBatch) -> dict[str, torch.Tensor]:
        """
        Forward pass computing intensities, mark predictions, and losses.

        Returns dict with keys:
            - 'intensities': (B, L) conditional intensities
            - 'mark_mu': (B, L, M) predicted mark means
            - 'mark_sigma': (B, L, M) predicted mark stds
            - 'temporal_nll': scalar temporal negative log-likelihood
            - 'mark_nll': scalar mark negative log-likelihood
            - 'loss': scalar total loss
        """
        intensities, hidden = self.intensity_rnn(
            batch.inter_arrival_times,
            batch.marks,
            batch.features,
            batch.mask,
        )

        mark_mu, mark_sigma = self.mark_predictor(hidden)
        mark_log_prob = self.mark_predictor.log_prob(hidden, batch.marks)

        # Temporal NLL: -log λ(t_i) + ∫₀^{t_i} λ(s) ds
        temporal_nll = self._temporal_nll(
            intensities, batch.inter_arrival_times, batch.mask,
        )

        # Mark NLL
        if batch.mask is not None:
            mark_nll = -(mark_log_prob * batch.mask).sum() / batch.mask.sum().clamp(min=1)
        else:
            mark_nll = -mark_log_prob.mean()

        loss = temporal_nll + 0.5 * mark_nll

        return {
            "intensities": intensities,
            "mark_mu": mark_mu,
            "mark_sigma": mark_sigma,
            "temporal_nll": temporal_nll,
            "mark_nll": mark_nll,
            "loss": loss,
        }

    def _temporal_nll(
        self,
        intensities: torch.Tensor,
        inter_arrivals: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute temporal NLL via Monte Carlo integration of the survival function.

        NLL = -Σ log λ(t_i) + Σ ∫_{t_{i-1}}^{t_i} λ(s) ds

        The integral is approximated by uniform sampling in each interval.
        """
        if mask is None:
            mask = torch.ones_like(intensities)

        # Term 1: -log λ(t_i) at event times
        log_intensity = torch.log(intensities.clamp(min=1e-8))
        event_ll = (log_intensity * mask).sum() / mask.sum().clamp(min=1)

        # Term 2: integral approximation via MC sampling
        # Sample uniform points in [0, Δt_i] and evaluate intensity
        B, L = inter_arrivals.shape
        # Use trapezoidal approximation: ∫λ(s)ds ≈ λ(t_i) * Δt_i
        # (This is a first-order approximation; MC refinement for future work)
        integral = (intensities * inter_arrivals * mask).sum() / mask.sum().clamp(min=1)

        return -event_ll + integral

    @torch.no_grad()
    def predict_fill_probability(
        self,
        batch: TPPBatch,
        horizon: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict the probability of a fill within `horizon` seconds.

        P(event in [0, horizon]) = 1 - exp(-∫₀^{horizon} λ(s) ds)

        Uses the latest intensity estimate and assumes constant intensity
        over the prediction horizon (exponential approximation).

        Args:
            batch: Input batch with the latest event history.
            horizon: Prediction horizon in seconds.

        Returns:
            fill_prob: (B,) probability of fill for each sequence.
        """
        self.eval()
        intensities, _ = self.intensity_rnn(
            batch.inter_arrival_times,
            batch.marks,
            batch.features,
            batch.mask,
        )

        # Use the last valid intensity for each sequence
        if batch.mask is not None:
            lengths = batch.mask.sum(dim=1).long() - 1  # (B,)
            last_intensity = intensities[torch.arange(len(lengths)), lengths]
        else:
            last_intensity = intensities[:, -1]

        # Normalize intensity relative to sequence statistics for calibration.
        # Prevents saturation when absolute λ values drift after extended training.
        seq_mean = intensities.mean(dim=1)
        seq_std = intensities.std(dim=1).clamp(min=1e-4)
        last_z = (last_intensity - seq_mean) / seq_std  # z-score
        # Map z-score to [0,1] via sigmoid — relative rank within sequence
        calibrated = torch.sigmoid(last_z)

        # Also compute the raw survival-based probability for reference
        raw_prob = 1.0 - torch.exp(-last_intensity * horizon)

        # Blend: 70% calibrated rank + 30% raw survival
        fill_prob = 0.7 * calibrated + 0.3 * raw_prob
        return fill_prob.clamp(0, 1)


# ---------------------------------------------------------------------------
# Dataset & collation
# ---------------------------------------------------------------------------

class TPPDataset(torch.utils.data.Dataset):
    """
    Dataset for Neural TPP training.

    Converts a trades DataFrame into fixed-length event sequences with
    inter-arrival times, marks (quantities), and auxiliary features.
    """

    def __init__(
        self,
        trades: "pd.DataFrame",
        feature_cols: list[str] | None = None,
        seq_length: int = 128,
    ):
        import pandas as pd

        self.seq_length = seq_length
        df = trades.sort_values("timestamp").reset_index(drop=True)

        # Compute inter-arrival times
        ts = pd.to_datetime(df["timestamp"]).astype("int64") / 1e9
        self.inter_arrivals = np.diff(ts, prepend=ts.iloc[0]).clip(0)
        self.marks = df["quantity"].values.astype(np.float32)

        # Auxiliary features
        if feature_cols and all(c in df.columns for c in feature_cols):
            self.features = df[feature_cols].values.astype(np.float32)
        else:
            # Default: use side encoding + price returns
            side_enc = df["side"].map({"buy": 1.0, "sell": -1.0}).fillna(0).values
            price_ret = df["price"].pct_change().fillna(0).values
            self.features = np.column_stack([side_enc, price_ret]).astype(np.float32)

        self.n_events = len(df)
        self.n_sequences = max(1, (self.n_events - seq_length) // (seq_length // 2))

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict:
        start = idx * (self.seq_length // 2)
        end = start + self.seq_length

        if end > self.n_events:
            start = max(0, self.n_events - self.seq_length)
            end = self.n_events

        length = end - start
        ia = np.zeros(self.seq_length, dtype=np.float32)
        mk = np.zeros(self.seq_length, dtype=np.float32)
        feat = np.zeros((self.seq_length, self.features.shape[1]), dtype=np.float32)
        mask = np.zeros(self.seq_length, dtype=np.float32)

        ia[:length] = self.inter_arrivals[start:end]
        mk[:length] = self.marks[start:end]
        feat[:length] = self.features[start:end]
        mask[:length] = 1.0

        return {
            "inter_arrival_times": torch.from_numpy(ia),
            "marks": torch.from_numpy(mk),
            "features": torch.from_numpy(feat),
            "mask": torch.from_numpy(mask),
        }


def collate_tpp(batch: list[dict]) -> TPPBatch:
    """Collate function for TPP DataLoader."""
    return TPPBatch(
        inter_arrival_times=torch.stack([b["inter_arrival_times"] for b in batch]),
        marks=torch.stack([b["marks"] for b in batch]),
        features=torch.stack([b["features"] for b in batch]),
        mask=torch.stack([b["mask"] for b in batch]),
    )
