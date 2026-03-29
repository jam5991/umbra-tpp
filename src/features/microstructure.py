"""
Market microstructure feature engineering.

Computes VPIN (Volume-synchronized Probability of Informed Trading),
Order Flow Imbalance (OFI), and volume-weighted microprice from
L2 order book snapshots and trade streams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vpin(
    trades: pd.DataFrame,
    bucket_size: float = 50.0,
    n_buckets: int | None = None,
) -> pd.Series:
    """
    Compute Volume-synchronized Probability of Informed Trading (VPIN).

    VPIN estimates the probability of informed trading by bucketing trades
    into equal-volume bars and measuring the imbalance between buy and
    sell volume in each bar.

    Args:
        trades: DataFrame with columns ['timestamp', 'price', 'quantity', 'side'].
        bucket_size: Volume per bucket (in base asset units, e.g. BTC).
        n_buckets: If set, override bucket_size to create exactly this many buckets.

    Returns:
        Series of VPIN values indexed by bucket end timestamp.
    """
    if trades.empty:
        return pd.Series(dtype=float, name="vpin")

    df = trades.sort_values("timestamp").reset_index(drop=True)
    df["signed_volume"] = df["quantity"] * df["side"].map({"buy": 1, "sell": -1}).fillna(0)

    total_volume = df["quantity"].sum()
    if n_buckets is not None and n_buckets > 0:
        bucket_size = total_volume / n_buckets

    # Assign rows to volume buckets
    df["cum_volume"] = df["quantity"].cumsum()
    df["bucket_id"] = (df["cum_volume"] / bucket_size).astype(int)

    # Aggregate by bucket
    buckets = df.groupby("bucket_id").agg(
        buy_volume=("signed_volume", lambda x: x[x > 0].sum()),
        sell_volume=("signed_volume", lambda x: abs(x[x < 0].sum())),
        total_volume=("quantity", "sum"),
        end_time=("timestamp", "last"),
    )

    # VPIN = |V_buy - V_sell| / V_total  (rolling average over all buckets)
    buckets["order_imbalance"] = np.abs(buckets["buy_volume"] - buckets["sell_volume"])
    buckets["vpin"] = buckets["order_imbalance"] / buckets["total_volume"].replace(0, np.nan)
    buckets["vpin"] = buckets["vpin"].fillna(0).clip(0, 1)

    result = buckets.set_index("end_time")["vpin"]
    result.name = "vpin"
    return result


def compute_ofi(
    depth: pd.DataFrame,
    n_levels: int = 5,
) -> pd.DataFrame:
    """
    Compute Order Flow Imbalance (OFI) from L2 order book snapshots.

    OFI measures the net pressure between bid and ask sides of the book
    across `n_levels` of depth.

    Args:
        depth: DataFrame with columns ['timestamp', 'price', 'quantity', 'side', 'snapshot_id'].
        n_levels: Number of price levels from each side to include.

    Returns:
        DataFrame with columns ['snapshot_id', 'ofi', 'bid_depth', 'ask_depth', 'depth_imbalance'].
    """
    if depth.empty:
        return pd.DataFrame(columns=["snapshot_id", "ofi", "bid_depth", "ask_depth", "depth_imbalance"])

    results = []
    for snap_id, snap in depth.groupby("snapshot_id"):
        bids = snap[snap["side"] == "bid"].nlargest(n_levels, "price")
        asks = snap[snap["side"] == "ask"].nsmallest(n_levels, "price")

        bid_depth = bids["quantity"].sum()
        ask_depth = asks["quantity"].sum()
        total_depth = bid_depth + ask_depth

        # OFI = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        ofi = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        # Depth imbalance at best level
        best_bid_qty = bids["quantity"].iloc[0] if len(bids) > 0 else 0.0
        best_ask_qty = asks["quantity"].iloc[0] if len(asks) > 0 else 0.0
        total_best = best_bid_qty + best_ask_qty
        depth_imbalance = (best_bid_qty - best_ask_qty) / total_best if total_best > 0 else 0.0

        results.append({
            "snapshot_id": snap_id,
            "ofi": ofi,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "depth_imbalance": depth_imbalance,
        })

    return pd.DataFrame(results)


def compute_microprice(
    depth: pd.DataFrame,
    n_levels: int = 5,
) -> pd.DataFrame:
    """
    Compute volume-weighted microprice from L2 snapshots.

    The microprice more accurately represents the "true" price by weighting
    the best bid and ask by their respective quantities. Extended across
    multiple levels for deeper signal.

    Args:
        depth: DataFrame with columns ['timestamp', 'price', 'quantity', 'side', 'snapshot_id'].
        n_levels: Number of bid/ask levels to weight.

    Returns:
        DataFrame with ['snapshot_id', 'microprice', 'midprice', 'spread_bps', 'weighted_spread'].
    """
    if depth.empty:
        return pd.DataFrame(columns=["snapshot_id", "microprice", "midprice", "spread_bps", "weighted_spread"])

    results = []
    for snap_id, snap in depth.groupby("snapshot_id"):
        bids = snap[snap["side"] == "bid"].nlargest(n_levels, "price")
        asks = snap[snap["side"] == "ask"].nsmallest(n_levels, "price")

        if bids.empty or asks.empty:
            continue

        # Volume-weighted microprice
        bid_vwap = (bids["price"] * bids["quantity"]).sum() / bids["quantity"].sum()
        ask_vwap = (asks["price"] * asks["quantity"]).sum() / asks["quantity"].sum()

        best_bid = bids["price"].iloc[0]
        best_ask = asks["price"].iloc[0]
        best_bid_qty = bids["quantity"].iloc[0]
        best_ask_qty = asks["quantity"].iloc[0]

        # Microprice: weighted by opposite-side volume (intuition: scarce side
        # has more "pull" on the price)
        total_best_qty = best_bid_qty + best_ask_qty
        microprice = (best_bid * best_ask_qty + best_ask * best_bid_qty) / total_best_qty

        midprice = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = (spread / midprice) * 10_000 if midprice > 0 else 0.0

        # Weighted spread (accounts for depth across levels)
        weighted_spread = ask_vwap - bid_vwap

        results.append({
            "snapshot_id": snap_id,
            "microprice": microprice,
            "midprice": midprice,
            "spread_bps": spread_bps,
            "weighted_spread": weighted_spread,
        })

    return pd.DataFrame(results)


def compute_trade_flow_features(trades: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Compute rolling trade-flow features from a trades DataFrame.

    Args:
        trades: DataFrame with ['timestamp', 'price', 'quantity', 'side'].
        window: Rolling window size (number of trades).

    Returns:
        DataFrame with columns for inter-arrival times, rolling volatility,
        trade intensity, and volume acceleration.
    """
    if trades.empty:
        return pd.DataFrame()

    df = trades.sort_values("timestamp").reset_index(drop=True)

    # Convert timestamps to seconds for inter-arrival times
    df["ts_seconds"] = pd.to_datetime(df["timestamp"]).astype("int64") / 1e9
    df["inter_arrival"] = df["ts_seconds"].diff().fillna(0).clip(lower=0)

    # Signed quantity
    df["signed_qty"] = df["quantity"] * df["side"].map({"buy": 1.0, "sell": -1.0}).fillna(0)

    # Rolling features
    df["rolling_volatility"] = df["price"].rolling(window, min_periods=1).std().fillna(0)
    df["rolling_volume"] = df["quantity"].rolling(window, min_periods=1).sum()
    df["rolling_net_flow"] = df["signed_qty"].rolling(window, min_periods=1).sum()
    df["trade_intensity"] = 1.0 / df["inter_arrival"].replace(0, np.nan).rolling(window, min_periods=1).mean().fillna(1)

    # Volume acceleration (change in rolling volume)
    df["volume_accel"] = df["rolling_volume"].diff().fillna(0)

    # Price momentum
    df["price_return"] = df["price"].pct_change().fillna(0)
    df["rolling_return"] = df["price_return"].rolling(window, min_periods=1).sum()

    return df


def build_feature_matrix(
    depth: pd.DataFrame,
    trades: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Orchestrate all feature computations into a single feature matrix.

    Combines VPIN, OFI, microprice, and trade-flow features. Each row
    corresponds to one snapshot/time-step with all features merged.

    Args:
        depth: L2 order book DataFrame.
        trades: Trades DataFrame.
        config: Optional config dict with keys from 'features' section.

    Returns:
        Feature matrix DataFrame ready for model input.
    """
    cfg = config or {}
    feat_cfg = cfg.get("features", {})

    bucket_size = feat_cfg.get("vpin_bucket_size", 50)
    ofi_depth = feat_cfg.get("ofi_depth", 5)
    micro_levels = feat_cfg.get("microprice_levels", 5)
    lookback = feat_cfg.get("lookback_window", 100)

    # Compute individual feature groups
    vpin = compute_vpin(trades, bucket_size=bucket_size)
    ofi_df = compute_ofi(depth, n_levels=ofi_depth)
    micro_df = compute_microprice(depth, n_levels=micro_levels)
    trade_features = compute_trade_flow_features(trades, window=lookback)

    # Merge OFI + microprice on snapshot_id
    features = ofi_df.copy()
    if not micro_df.empty:
        features = features.merge(micro_df, on="snapshot_id", how="outer")

    # Attach VPIN as a scalar feature (use mean VPIN per snapshot period)
    if not vpin.empty:
        features["vpin"] = vpin.mean()  # Scalar across this data slice

    # Attach aggregated trade-flow features
    if not trade_features.empty:
        agg_cols = ["rolling_volatility", "trade_intensity", "rolling_net_flow",
                    "volume_accel", "rolling_return", "inter_arrival"]
        for col in agg_cols:
            if col in trade_features.columns:
                features[col] = trade_features[col].iloc[-1]  # Latest value

    features = features.fillna(0)
    return features
