"""
Fetch real market data from Binance and OKX public APIs.

Retrieves L2 order book snapshots and recent trades for BTC pairs,
then saves them as CSV files for downstream feature engineering.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binance (auto-fallback: api.binance.com → api.binance.us)
# ---------------------------------------------------------------------------

# Binance base URL with automatic geo-restriction fallback.
# Tries api.binance.com first; if it returns 451, falls back to api.binance.us
# and caches the working URL for all subsequent requests.
_BINANCE_BASE_URLS = ["https://api.binance.com", "https://api.binance.us"]
_binance_base_url: str | None = None  # Cached after first successful request


def _binance_get(path: str, params: dict | None = None) -> dict:
    """
    Make a GET request to the Binance API with automatic US fallback.

    Tries api.binance.com first. If it returns HTTP 451 (geo-restricted),
    retries with api.binance.us and caches the working base URL.
    """
    global _binance_base_url

    # If we already know which base URL works, use it directly
    if _binance_base_url is not None:
        url = f"{_binance_base_url}{path}"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # First call: try each base URL in order
    last_error = None
    for base_url in _BINANCE_BASE_URLS:
        url = f"{base_url}{path}"
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 451:
                logger.info(f"Binance {base_url} returned 451 (geo-restricted), trying next...")
                continue
            resp.raise_for_status()
            # Success — cache this base URL for future calls
            _binance_base_url = base_url
            logger.info(f"Using Binance endpoint: {base_url}")
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 451:
                logger.info(f"Binance {base_url} returned 451 (geo-restricted), trying next...")
                last_error = e
                continue
            raise
        except Exception as e:
            last_error = e
            logger.warning(f"Binance {base_url} failed: {e}, trying next...")
            continue

    raise RuntimeError(
        f"All Binance endpoints failed. Last error: {last_error}"
    )


def fetch_binance_depth(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """Fetch L2 order book snapshot from Binance."""
    data = _binance_get("/api/v3/depth", {"symbol": symbol, "limit": limit})

    rows = []
    ts = datetime.now(timezone.utc).isoformat()
    for price, qty in data["bids"]:
        rows.append({"timestamp": ts, "price": float(price), "quantity": float(qty),
                      "side": "bid", "venue": "binance", "symbol": symbol})
    for price, qty in data["asks"]:
        rows.append({"timestamp": ts, "price": float(price), "quantity": float(qty),
                      "side": "ask", "venue": "binance", "symbol": symbol})
    return pd.DataFrame(rows)


def fetch_binance_trades(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """Fetch recent trades from Binance."""
    data = _binance_get("/api/v3/trades", {"symbol": symbol, "limit": limit})

    rows = []
    for t in data:
        rows.append({
            "trade_id": t["id"],
            "timestamp": datetime.fromtimestamp(t["time"] / 1000, tz=timezone.utc).isoformat(),
            "price": float(t["price"]),
            "quantity": float(t["qty"]),
            "side": "sell" if t["isBuyerMaker"] else "buy",
            "venue": "binance",
            "symbol": symbol,
        })
    return pd.DataFrame(rows)


def fetch_binance_agg_trades(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """Fetch aggregated trades from Binance (compressed trade stream)."""
    data = _binance_get("/api/v3/aggTrades", {"symbol": symbol, "limit": limit})

    rows = []
    for t in data:
        rows.append({
            "agg_trade_id": t["a"],
            "timestamp": datetime.fromtimestamp(t["T"] / 1000, tz=timezone.utc).isoformat(),
            "price": float(t["p"]),
            "quantity": float(t["q"]),
            "side": "sell" if t["m"] else "buy",
            "venue": "binance",
            "symbol": symbol,
            "first_trade_id": t["f"],
            "last_trade_id": t["l"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OKX
# ---------------------------------------------------------------------------

def fetch_okx_depth(symbol: str = "BTC-USDT", limit: int = 400) -> pd.DataFrame:
    """Fetch L2 order book snapshot from OKX."""
    url = f"https://www.okx.com/api/v5/market/books?instId={symbol}&sz={limit}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data["code"] != "0":
        raise RuntimeError(f"OKX API error: {data['msg']}")

    rows = []
    ts = datetime.now(timezone.utc).isoformat()
    book = data["data"][0]
    for entry in book["bids"]:
        price, qty = float(entry[0]), float(entry[1])
        rows.append({"timestamp": ts, "price": price, "quantity": qty,
                      "side": "bid", "venue": "okx", "symbol": symbol})
    for entry in book["asks"]:
        price, qty = float(entry[0]), float(entry[1])
        rows.append({"timestamp": ts, "price": price, "quantity": qty,
                      "side": "ask", "venue": "okx", "symbol": symbol})
    return pd.DataFrame(rows)


def fetch_okx_trades(symbol: str = "BTC-USDT", limit: int = 500) -> pd.DataFrame:
    """Fetch recent trades from OKX."""
    url = f"https://www.okx.com/api/v5/market/trades?instId={symbol}&limit={limit}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data["code"] != "0":
        raise RuntimeError(f"OKX API error: {data['msg']}")

    rows = []
    for t in data["data"]:
        rows.append({
            "trade_id": t["tradeId"],
            "timestamp": datetime.fromtimestamp(int(t["ts"]) / 1000, tz=timezone.utc).isoformat(),
            "price": float(t["px"]),
            "quantity": float(t["sz"]),
            "side": t["side"],
            "venue": "okx",
            "symbol": symbol,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-snapshot collection
# ---------------------------------------------------------------------------

def collect_snapshots(
    n_snapshots: int = 10,
    interval_sec: float = 2.0,
    venues: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect multiple L2 snapshots and trade batches over time.

    Returns:
        (depth_df, trades_df) — concatenated DataFrames across snapshots.
    """
    if venues is None:
        venues = ["binance", "okx"]

    all_depth = []
    all_trades = []

    for i in range(n_snapshots):
        logger.info(f"Snapshot {i + 1}/{n_snapshots}")

        if "binance" in venues:
            try:
                depth = fetch_binance_depth()
                depth["snapshot_id"] = i
                all_depth.append(depth)

                trades = fetch_binance_trades()
                trades["snapshot_id"] = i
                all_trades.append(trades)
            except Exception as e:
                logger.warning(f"Binance fetch failed: {e}")

        if "okx" in venues:
            try:
                depth = fetch_okx_depth()
                depth["snapshot_id"] = i
                all_depth.append(depth)

                trades = fetch_okx_trades()
                trades["snapshot_id"] = i
                all_trades.append(trades)
            except Exception as e:
                logger.warning(f"OKX fetch failed: {e}")

        if i < n_snapshots - 1:
            time.sleep(interval_sec)

    depth_df = pd.concat(all_depth, ignore_index=True) if all_depth else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # Deduplicate trades by trade_id + venue
    if not trades_df.empty and "trade_id" in trades_df.columns:
        trades_df = trades_df.drop_duplicates(subset=["trade_id", "venue"]).reset_index(drop=True)

    return depth_df, trades_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch real market data from Binance/OKX")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--snapshots", type=int, default=10,
                        help="Number of order book snapshots to collect")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between snapshots")
    parser.add_argument("--output-dir", type=str, default="data/l2_snapshots",
                        help="Output directory for CSV files")
    parser.add_argument("--venues", nargs="+", default=["binance", "okx"],
                        help="Venues to fetch from")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting {args.snapshots} snapshots from {args.venues}...")
    depth_df, trades_df = collect_snapshots(
        n_snapshots=args.snapshots,
        interval_sec=args.interval,
        venues=args.venues,
    )

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not depth_df.empty:
        depth_path = out_dir / f"depth_{ts_tag}.csv"
        depth_df.to_csv(depth_path, index=False)
        logger.info(f"Saved {len(depth_df)} depth rows → {depth_path}")
    else:
        logger.warning("No depth data collected")

    if not trades_df.empty:
        trades_path = out_dir / f"trades_{ts_tag}.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved {len(trades_df)} trade rows → {trades_path}")
    else:
        logger.warning("No trade data collected")

    # Print summary
    logger.info("--- Summary ---")
    if not depth_df.empty:
        for venue in depth_df["venue"].unique():
            v_df = depth_df[depth_df["venue"] == venue]
            logger.info(f"  {venue}: {len(v_df)} depth levels, "
                        f"mid={v_df['price'].median():.2f}")
    if not trades_df.empty:
        for venue in trades_df["venue"].unique():
            v_df = trades_df[trades_df["venue"] == venue]
            logger.info(f"  {venue}: {len(v_df)} trades, "
                        f"avg_price={v_df['price'].mean():.2f}, "
                        f"total_vol={v_df['quantity'].sum():.4f} BTC")


if __name__ == "__main__":
    main()
