#!/usr/bin/env python3
"""
Umbra-TPP Training Script.

Loads market data, builds features, trains the NeuralTPP model,
runs causal estimation, and optionally runs a backtest.

Usage:
    python scripts/train_tpp.py --config configs/default.yaml
    python scripts/train_tpp.py --config configs/default.yaml --epochs 10 --backtest
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.tpp_core import NeuralTPP, TPPDataset, collate_tpp, TPPBatch
from src.model.causal_layer import (
    MarketImpactEstimator,
    AdversarialDebias,
    estimate_fill_probability,
)
from src.features.microstructure import build_feature_matrix
from src.backtest.engine import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_tpp")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the most recent depth and trades CSVs from data_dir."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    depth_files = sorted(data_path.glob("depth_*.csv"))
    trade_files = sorted(data_path.glob("trades_*.csv"))

    if not trade_files:
        raise FileNotFoundError(f"No trade files found in {data_dir}")

    # Load the most recent files
    logger.info(f"Loading trades from {trade_files[-1]}")
    trades = pd.read_csv(trade_files[-1])

    depth = pd.DataFrame()
    if depth_files:
        logger.info(f"Loading depth from {depth_files[-1]}")
        depth = pd.read_csv(depth_files[-1])

    logger.info(f"Loaded {len(trades)} trades, {len(depth)} depth rows")
    return depth, trades


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: NeuralTPP,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None,
    config: dict,
) -> dict:
    """
    Train the NeuralTPP model.

    Returns:
        Training history dict with loss curves.
    """
    train_cfg = config.get("training", {})
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    epochs = train_cfg.get("epochs", 50)
    patience = train_cfg.get("patience", 10)
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2,
    )

    history = {
        "train_loss": [],
        "train_temporal_nll": [],
        "train_mark_nll": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        epoch_temporal = 0.0
        epoch_mark = 0.0
        n_batches = 0

        for batch_dict in train_loader:
            batch = collate_tpp([batch_dict]) if isinstance(batch_dict, dict) else batch_dict

            output = model(batch)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_temporal += output["temporal_nll"].item()
            epoch_mark += output["mark_nll"].item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_temporal = epoch_temporal / max(n_batches, 1)
        avg_mark = epoch_mark / max(n_batches, 1)

        history["train_loss"].append(avg_loss)
        history["train_temporal_nll"].append(avg_temporal)
        history["train_mark_nll"].append(avg_mark)

        # --- Validation ---
        val_loss = float("inf")
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for batch_dict in val_loader:
                    batch = collate_tpp([batch_dict]) if isinstance(batch_dict, dict) else batch_dict
                    output = model(batch)
                    val_total += output["loss"].item()
                    val_n += 1
            val_loss = val_total / max(val_n, 1)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

        # --- Logging ---
        log_msg = (
            f"Epoch {epoch + 1:3d}/{epochs} │ "
            f"loss={avg_loss:.4f} (temporal={avg_temporal:.4f}, mark={avg_mark:.4f})"
        )
        if val_loader is not None:
            log_msg += f" │ val_loss={val_loss:.4f}"
        log_msg += f" │ lr={optimizer.param_groups[0]['lr']:.6f}"
        logger.info(log_msg)

        # --- Early stopping & checkpointing ---
        check_loss = val_loss if val_loader is not None else avg_loss
        if check_loss < best_val_loss:
            best_val_loss = check_loss
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": check_loss,
            }, checkpoint_path)
            logger.info(f"  ✓ Saved checkpoint → {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                break

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
        "config": config,
    }, final_path)
    logger.info(f"Saved final model → {final_path}")

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Umbra-TPP model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data/l2_snapshots")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest after training")
    parser.add_argument("--no-causal", action="store_true",
                        help="Skip causal estimation")
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    logger.info("=" * 60)
    logger.info("  Umbra-TPP Training Pipeline")
    logger.info("=" * 60)

    # ━━━ Step 1: Load Data ━━━
    logger.info("\n━━━ Step 1: Loading Market Data ━━━")
    data_dir = PROJECT_ROOT / args.data_dir
    depth, trades = load_data(str(data_dir))

    # ━━━ Step 2: Build Features ━━━
    logger.info("\n━━━ Step 2: Building Feature Matrix ━━━")
    feature_matrix = build_feature_matrix(depth, trades, config)
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    logger.info(f"Features: {list(feature_matrix.columns)}")

    # ━━━ Step 3: Prepare Dataset ━━━
    logger.info("\n━━━ Step 3: Preparing Dataset ━━━")
    seq_length = config["training"].get("sequence_length", 128)
    dataset = TPPDataset(trades, seq_length=seq_length)
    logger.info(f"Dataset: {len(dataset)} sequences of length {seq_length}")

    # Train/val split
    train_split = config["training"].get("train_split", 0.8)
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train

    if n_val > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = dataset
        val_dataset = None

    batch_size = min(config["training"].get("batch_size", 64), len(train_dataset))
    batch_size = max(batch_size, 1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_tpp, drop_last=False,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) >= batch_size:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_tpp,
        )

    # ━━━ Step 4: Build Model ━━━
    logger.info("\n━━━ Step 4: Building NeuralTPP Model ━━━")
    feature_dim = dataset.features.shape[1]  # From the underlying dataset
    model_cfg = config.get("model", {})

    model = NeuralTPP(
        feature_dim=feature_dim,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        mark_dim=model_cfg.get("mark_dim", 1),
        dropout=model_cfg.get("dropout", 0.1),
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Architecture:\n{model}")

    # ━━━ Step 5: Train ━━━
    logger.info("\n━━━ Step 5: Training ━━━")
    history = train(model, train_loader, val_loader, config)

    # Print loss summary
    logger.info("\n━━━ Training Complete ━━━")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        logger.info(f"  Final val loss:   {history['val_loss'][-1]:.4f}")
    logger.info(f"  Best loss:        {min(history['train_loss']):.4f}")

    # ━━━ Step 6: Causal Estimation ━━━
    if not args.no_causal:
        logger.info("\n━━━ Step 6: Causal Market Impact Estimation ━━━")

        # Build a causal dataset with simulated treatment/outcome.
        # In production, treatment = whether a dark pool order was placed,
        # outcome = observed price impact in bps.
        causal_df = trades.copy().sort_values("timestamp").reset_index(drop=True)
        causal_df["is_dark_fill"] = (causal_df["quantity"] > causal_df["quantity"].median()).astype(float)
        causal_df["price_impact_bps"] = causal_df["price"].pct_change().fillna(0) * 10_000

        # Rich confounders — more features → better nuisance R²
        causal_df["volatility"] = causal_df["price"].rolling(20, min_periods=1).std().fillna(0)
        causal_df["volume_flow"] = causal_df["quantity"].rolling(20, min_periods=1).sum().fillna(0)
        causal_df["rolling_return"] = causal_df["price"].pct_change().fillna(0).rolling(10, min_periods=1).sum()
        causal_df["price_momentum"] = causal_df["price"].pct_change(5).fillna(0) * 10_000
        causal_df["volume_accel"] = causal_df["volume_flow"].diff().fillna(0)
        causal_df["trade_intensity"] = 1.0 / (
            pd.to_datetime(causal_df["timestamp"]).astype("int64").diff().fillna(1e9) / 1e9
        ).clip(lower=1e-3).rolling(10, min_periods=1).mean()
        causal_df["signed_flow"] = (
            causal_df["quantity"] * causal_df["side"].map({"buy": 1.0, "sell": -1.0}).fillna(0)
        ).rolling(10, min_periods=1).sum()
        causal_df["abs_impact_lag1"] = causal_df["price_impact_bps"].abs().shift(1).fillna(0)

        confounder_cols = [
            "volatility", "volume_flow", "rolling_return", "price_momentum",
            "volume_accel", "trade_intensity", "signed_flow", "abs_impact_lag1",
        ]

        estimator = MarketImpactEstimator(
            n_folds=config["causal"].get("n_folds", 3),
            nuisance_model=config["causal"].get("nuisance_model", "gradient_boosting"),
        )

        result = estimator.estimate(
            causal_df,
            treatment_col="is_dark_fill",
            outcome_col="price_impact_bps",
            confounder_cols=confounder_cols,
        )
        logger.info(f"  ATE = {result.ate:.4f} ± {result.ate_std:.4f} bps")
        logger.info(f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        logger.info(f"  Nuisance R² (treatment): {result.nuisance_r2_treatment:.3f}")
        logger.info(f"  Nuisance R² (outcome):   {result.nuisance_r2_outcome:.3f}")

        # ━━━ Adversarial Debiasing ━━━
        logger.info("\n━━━ Adversarial Debiasing ━━━")
        adversarial_cfg = config["causal"]
        features_np = causal_df[confounder_cols].fillna(0).values
        labels_np = causal_df["is_dark_fill"].values

        debias = AdversarialDebias(feature_dim=len(confounder_cols))
        debias_losses = debias.train_discriminator(
            features_np, labels_np,
            epochs=adversarial_cfg.get("adversarial_epochs", 20),
            lr=adversarial_cfg.get("adversarial_lr", 0.0005),
        )
        logger.info(f"  Debiasing loss: {debias_losses[0]:.4f} → {debias_losses[-1]:.4f}")

        # Compute adjusted fill probabilities (demo)
        model.eval()
        n_demo = min(len(dataset), batch_size)
        demo_items = [dataset[i] for i in range(n_demo)]
        sample_batch = collate_tpp(demo_items)
        tpp_probs = model.predict_fill_probability(sample_batch).numpy()

        weights = debias.compute_selection_weights(
            torch.from_numpy(features_np[:len(tpp_probs)].astype(np.float32))
        ).numpy()

        adjusted_probs = estimate_fill_probability(
            tpp_probs, result.ate, weights
        )
        logger.info(f"  TPP fill prob:      mean={tpp_probs.mean():.4f}, std={tpp_probs.std():.4f}")
        logger.info(f"  Adjusted fill prob: mean={adjusted_probs.mean():.4f}, std={adjusted_probs.std():.4f}")

    # ━━━ Step 7: Backtest ━━━
    if args.backtest:
        logger.info("\n━━━ Step 7: Backtesting ━━━")
        metrics = run_backtest(model, trades, config)
        print(metrics.summary())

    logger.info("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
