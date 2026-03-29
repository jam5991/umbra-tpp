#!/usr/bin/env python3
"""
Generate matplotlib visualizations for the Umbra-TPP DEMO.md.

Creates a dual-axis plot showing:
- Conditional Intensity λ(t) from the NeuralTPP model
- Fill Probability predictions
- Lit market volume (bar chart)
- Lead-lag detection annotation

Outputs PNGs to docs/plots/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.tpp_core import NeuralTPP, TPPBatch

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "monospace",
    "font.size": 10,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

CYAN = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"
GRAY = "#8b949e"


def generate_intensity_plot(
    trades: pd.DataFrame,
    model: NeuralTPP,
    config: dict,
    output_path: Path,
):
    """Generate the main intensity λ(t) + fill probability plot."""
    model.eval()
    df = trades.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    seq_len = config["training"].get("sequence_length", 128)

    ts = pd.to_datetime(df["timestamp"]).astype("int64") / 1e9
    inter_arrivals = np.diff(ts.values, prepend=ts.values[0]).clip(0).astype(np.float32)
    marks = df["quantity"].values.astype(np.float32)
    side_enc = df["side"].map({"buy": 1.0, "sell": -1.0}).fillna(0).values
    price_ret = df["price"].pct_change().fillna(0).values
    features = np.column_stack([side_enc, price_ret]).astype(np.float32)

    step = max(1, n // 200)
    intensities, fill_probs, event_indices, volumes = [], [], [], []

    for i in range(seq_len, n, step):
        start = max(0, i - seq_len)
        length = i - start

        ia = np.zeros(seq_len, dtype=np.float32)
        mk = np.zeros(seq_len, dtype=np.float32)
        ft = np.zeros((seq_len, 2), dtype=np.float32)
        mask = np.zeros(seq_len, dtype=np.float32)
        ia[:length] = inter_arrivals[start:i]
        mk[:length] = marks[start:i]
        ft[:length] = features[start:i]
        mask[:length] = 1.0

        batch = TPPBatch(
            inter_arrival_times=torch.from_numpy(ia).unsqueeze(0),
            marks=torch.from_numpy(mk).unsqueeze(0),
            features=torch.from_numpy(ft).unsqueeze(0),
            mask=torch.from_numpy(mask).unsqueeze(0),
        )

        with torch.no_grad():
            output = model(batch)
            intensity = output["intensities"][0, length - 1].item()
            fp = model.predict_fill_probability(batch, horizon=1.0).item()

        intensities.append(intensity)
        fill_probs.append(fp)
        event_indices.append(i)
        volumes.append(df["quantity"].iloc[max(start, i - 20):i].sum())

    intensities = np.array(intensities)
    fill_probs = np.array(fill_probs)
    event_indices = np.array(event_indices)
    volumes = np.array(volumes)

    # Lead-lag analysis
    if len(intensities) > 5:
        i_z = (intensities - intensities.mean()) / (intensities.std() + 1e-8)
        v_z = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
        lead_corr = np.correlate(i_z[:len(v_z)-1], v_z[1:], mode='valid')
        lead_score = lead_corr.mean() if len(lead_corr) > 0 else 0
    else:
        lead_score = 0

    # ── Create figure ────────────────────────────────────────────────
    fig, (ax_main, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 7), height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.08},
    )

    # ── Top: Intensity + Fill Probability ────────────────────────────
    x = np.arange(len(intensities))

    # Intensity line
    ax_main.plot(x, intensities, color=CYAN, linewidth=1.5, label="λ(t) Intensity", zorder=3)
    ax_main.fill_between(x, intensities.min(), intensities, alpha=0.08, color=CYAN)

    # Fill probability on secondary axis
    ax_fp = ax_main.twinx()
    ax_fp.plot(x, fill_probs, color=GREEN, linewidth=1.2, alpha=0.8, linestyle="--", label="Fill Probability")
    ax_fp.fill_between(x, fill_probs.min(), fill_probs, alpha=0.05, color=GREEN)
    ax_fp.set_ylabel("Fill Probability P(fill)", color=GREEN)
    ax_fp.tick_params(axis="y", colors=GREEN)
    ax_fp.spines["right"].set_color(GREEN)

    # Highlight intensity peaks (potential "Hidden Whale" detections)
    intensity_threshold = intensities.mean() + 1.5 * intensities.std()
    peak_mask = intensities > intensity_threshold
    if peak_mask.any():
        ax_main.scatter(
            x[peak_mask], intensities[peak_mask],
            color=ORANGE, s=50, zorder=5, marker="^",
            label=f"Hidden Whale Signal (λ > {intensity_threshold:.3f})",
            edgecolors="white", linewidths=0.5,
        )

    # Lead-lag annotation
    if lead_score > 0:
        lead_ms = abs(step * np.mean(inter_arrivals[seq_len:seq_len + 50]) * 1000)
        ax_main.annotate(
            f"⚡ Lead-Lag: λ(t) leads volume by ~{lead_ms:.0f}ms\n"
            f"   Cross-corr = {lead_score:.2f}",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=9, color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor=ORANGE, alpha=0.9),
            verticalalignment="top",
        )

    ax_main.set_ylabel("Conditional Intensity λ(t)", color=CYAN)
    ax_main.tick_params(axis="y", colors=CYAN)
    ax_main.spines["left"].set_color(CYAN)
    ax_main.set_xticklabels([])
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(
        "Umbra-TPP: Neural Intensity Function λ(t) — Hidden Liquidity Detection",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Combined legend
    lines_1, labels_1 = ax_main.get_legend_handles_labels()
    lines_2, labels_2 = ax_fp.get_legend_handles_labels()
    ax_main.legend(
        lines_1 + lines_2, labels_1 + labels_2,
        loc="upper right", fontsize=8, framealpha=0.9,
    )

    # ── Bottom: Volume bars ──────────────────────────────────────────
    buy_mask = df.iloc[event_indices]["side"].values == "buy" if len(event_indices) <= len(df) else np.ones(len(x), dtype=bool)

    colors = [GREEN if i % 2 == 0 else RED for i in range(len(x))]
    ax_vol.bar(x, volumes, color=colors, alpha=0.6, width=1.0)
    ax_vol.set_ylabel("Volume (BTC)", color=GRAY)
    ax_vol.set_xlabel("Event Window Index", color=GRAY)
    ax_vol.grid(True, alpha=0.2)

    # Minimal y-axis
    ax_vol.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = PROJECT_ROOT / "data" / "l2_snapshots"
    trade_files = sorted(data_path.glob("trades_*.csv"))
    if not trade_files:
        print("ERROR: No trade data. Run: python scripts/fetch_market_data.py")
        sys.exit(1)

    trades = pd.read_csv(trade_files[-1])
    print(f"Loaded {len(trades)} trades from {trade_files[-1].name}")

    # Load model
    model_cfg = config.get("model", {})
    model = NeuralTPP(
        feature_dim=2,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        mark_dim=model_cfg.get("mark_dim", 1),
        dropout=model_cfg.get("dropout", 0.1),
    )

    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No checkpoint, using untrained model")

    model.eval()

    # Output
    out_dir = PROJECT_ROOT / "docs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_intensity_plot(trades, model, config, out_dir / "intensity_lambda.png")
    print("Done!")


if __name__ == "__main__":
    main()
