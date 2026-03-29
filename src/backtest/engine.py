"""
Event-driven dark pool backtest simulator.

Replays a historical event stream, uses a trained NeuralTPP model to
predict fill probability at each decision point, and simulates order
placement and execution in a dark pool setting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from src.model.tpp_core import NeuralTPP, TPPBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Aggregated backtest performance metrics."""
    total_orders: int = 0
    filled_orders: int = 0
    fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    total_slippage_bps: float = 0.0
    avg_market_impact_bps: float = 0.0
    total_volume_btc: float = 0.0
    total_pnl_usd: float = 0.0
    avg_fill_time_sec: float = 0.0
    sharpe_ratio: float = 0.0

    # Per-order details
    order_log: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"═══ Backtest Summary ═══\n"
            f"  Orders:        {self.total_orders}\n"
            f"  Fills:         {self.filled_orders} ({self.fill_rate:.1%})\n"
            f"  Avg Slippage:  {self.avg_slippage_bps:.2f} bps\n"
            f"  Market Impact: {self.avg_market_impact_bps:.2f} bps\n"
            f"  Volume:        {self.total_volume_btc:.4f} BTC\n"
            f"  PnL:           ${self.total_pnl_usd:,.2f}\n"
            f"  Avg Fill Time: {self.avg_fill_time_sec:.2f}s\n"
            f"  Sharpe Ratio:  {self.sharpe_ratio:.3f}\n"
            f"════════════════════════"
        )


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------

def slippage_linear(quantity: float, base_bps: float = 0.5) -> float:
    """Linear slippage: slippage grows linearly with order size."""
    return base_bps * quantity


def slippage_sqrt(quantity: float, base_bps: float = 0.5) -> float:
    """Square-root slippage: more realistic for large orders."""
    return base_bps * np.sqrt(quantity)


SLIPPAGE_MODELS = {
    "linear": slippage_linear,
    "sqrt": slippage_sqrt,
}


# ---------------------------------------------------------------------------
# Dark Pool Simulator
# ---------------------------------------------------------------------------

class DarkPoolSimulator:
    """
    Event-driven simulator that replays market events and uses the
    NeuralTPP model to decide when to place dark pool orders.

    At each decision point, the simulator:
    1. Feeds event history into the TPP model
    2. Gets a fill probability prediction
    3. If P(fill) > threshold, places an order
    4. Simulates the fill based on actual market conditions
    """

    def __init__(
        self,
        model: NeuralTPP,
        config: dict,
    ):
        self.model = model
        self.model.eval()

        bt_cfg = config.get("backtest", {})
        self.initial_capital = bt_cfg.get("initial_capital_usd", 1_000_000)
        self.block_size = bt_cfg.get("block_size_btc", 1.0)
        self.fill_threshold = bt_cfg.get("fill_threshold", 0.6)
        self.warmup_events = bt_cfg.get("warmup_events", 200)

        slippage_type = bt_cfg.get("slippage_model", "sqrt")
        self.slippage_fn = SLIPPAGE_MODELS.get(slippage_type, slippage_sqrt)
        self.slippage_base_bps = bt_cfg.get("slippage_base_bps", 0.5)

        # Execution delay: simulates the latency between signal and fill.
        # Even 1ms matters at HFT scale — this makes PnL more realistic.
        self.execution_delay_ms: float = bt_cfg.get("execution_delay_ms", 1.0)

        # Explicit transaction cost (exchange fees, clearing, etc.)
        self.transaction_cost_bps: float = bt_cfg.get("transaction_cost_bps", 0.1)

        self.seq_length = config.get("training", {}).get("sequence_length", 128)

    def _build_batch(
        self,
        inter_arrivals: np.ndarray,
        marks: np.ndarray,
        features: np.ndarray,
    ) -> TPPBatch:
        """Construct a TPPBatch from numpy arrays (single sequence)."""
        L = min(len(inter_arrivals), self.seq_length)

        ia = np.zeros(self.seq_length, dtype=np.float32)
        mk = np.zeros(self.seq_length, dtype=np.float32)
        ft = np.zeros((self.seq_length, features.shape[-1]), dtype=np.float32)
        mask = np.zeros(self.seq_length, dtype=np.float32)

        ia[:L] = inter_arrivals[-L:]
        mk[:L] = marks[-L:]
        ft[:L] = features[-L:]
        mask[:L] = 1.0

        return TPPBatch(
            inter_arrival_times=torch.from_numpy(ia).unsqueeze(0),
            marks=torch.from_numpy(mk).unsqueeze(0),
            features=torch.from_numpy(ft).unsqueeze(0),
            mask=torch.from_numpy(mask).unsqueeze(0),
        )

    def run(self, trades: pd.DataFrame) -> BacktestMetrics:
        """
        Run the backtest on a trade history.

        Args:
            trades: DataFrame with ['timestamp', 'price', 'quantity', 'side'].

        Returns:
            BacktestMetrics with full results.
        """
        df = trades.sort_values("timestamp").reset_index(drop=True)
        n = len(df)

        if n < self.warmup_events + 10:
            logger.warning(f"Only {n} events; need at least {self.warmup_events + 10}")
            return BacktestMetrics()

        # Precompute arrays
        ts = pd.to_datetime(df["timestamp"]).astype("int64") / 1e9
        inter_arrivals = np.diff(ts.values, prepend=ts.values[0]).clip(0).astype(np.float32)
        marks = df["quantity"].values.astype(np.float32)
        side_enc = df["side"].map({"buy": 1.0, "sell": -1.0}).fillna(0).values
        price_ret = df["price"].pct_change().fillna(0).values
        features = np.column_stack([side_enc, price_ret]).astype(np.float32)

        prices = df["price"].values

        # Simulation state
        capital = self.initial_capital
        order_log = []
        pnl_series = []

        decision_interval = max(10, self.warmup_events // 10)

        for i in range(self.warmup_events, n, decision_interval):
            # Build input from history
            start = max(0, i - self.seq_length)
            batch = self._build_batch(
                inter_arrivals[start:i],
                marks[start:i],
                features[start:i],
            )

            # Predict fill probability
            fill_prob = self.model.predict_fill_probability(batch, horizon=1.0)
            fill_prob = fill_prob.item()

            current_price = prices[i]

            if fill_prob >= self.fill_threshold:
                # Place order — apply execution delay
                # In production, the delay between signal and fill means the
                # price can move. We model this by looking ahead by delay_events.
                delay_events = max(1, int(self.execution_delay_ms / (
                    np.mean(inter_arrivals[max(0, i-20):i]) * 1000 + 1e-6
                )))
                delayed_idx = min(i + delay_events, n - 1)
                execution_price = prices[delayed_idx]  # Price after delay

                slippage_bps = self.slippage_fn(self.block_size, self.slippage_base_bps)
                fill_price = execution_price * (1 + (slippage_bps + self.transaction_cost_bps) / 10_000)
                cost = fill_price * self.block_size

                # Simulate fill: probabilistic based on fill_prob
                rng = np.random.default_rng(seed=i)
                actually_filled = rng.random() < fill_prob

                if actually_filled and cost <= capital:
                    # PnL: compare fill price to future reference
                    future_idx = min(i + decision_interval, n - 1)
                    future_price = prices[future_idx]
                    pnl = (future_price - fill_price) * self.block_size

                    capital += pnl
                    pnl_series.append(pnl)

                    # Market impact: difference between fill price and pre-delay price
                    impact_bps = (fill_price / current_price - 1) * 10_000

                    order_log.append({
                        "event_idx": i,
                        "timestamp": df["timestamp"].iloc[i],
                        "fill_prob": fill_prob,
                        "current_price": current_price,
                        "execution_price": execution_price,
                        "fill_price": fill_price,
                        "slippage_bps": slippage_bps,
                        "transaction_cost_bps": self.transaction_cost_bps,
                        "execution_delay_ms": self.execution_delay_ms,
                        "impact_bps": impact_bps,
                        "pnl": pnl,
                        "filled": True,
                        "block_size": self.block_size,
                    })
                else:
                    order_log.append({
                        "event_idx": i,
                        "timestamp": df["timestamp"].iloc[i],
                        "fill_prob": fill_prob,
                        "current_price": current_price,
                        "filled": False,
                        "reason": "insufficient_capital" if cost > capital else "sim_miss",
                    })

        # Compute metrics
        metrics = BacktestMetrics(order_log=order_log)
        if order_log:
            metrics.total_orders = len(order_log)
            filled = [o for o in order_log if o.get("filled", False)]
            metrics.filled_orders = len(filled)
            metrics.fill_rate = len(filled) / len(order_log) if order_log else 0

            if filled:
                metrics.avg_slippage_bps = np.mean([o["slippage_bps"] for o in filled])
                metrics.total_slippage_bps = np.sum([o["slippage_bps"] for o in filled])
                metrics.avg_market_impact_bps = np.mean([o["impact_bps"] for o in filled])
                metrics.total_volume_btc = sum(o["block_size"] for o in filled)

            if pnl_series:
                metrics.total_pnl_usd = sum(pnl_series)
                pnl_arr = np.array(pnl_series)
                if pnl_arr.std() > 0:
                    metrics.sharpe_ratio = (pnl_arr.mean() / pnl_arr.std()) * np.sqrt(252)

        return metrics


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_backtest(
    model: NeuralTPP,
    trades: pd.DataFrame,
    config: dict,
) -> BacktestMetrics:
    """
    Run a full backtest simulation.

    Args:
        model: Trained NeuralTPP model.
        trades: Historical trades DataFrame.
        config: Full config dict.

    Returns:
        BacktestMetrics with backtest results.
    """
    sim = DarkPoolSimulator(model=model, config=config)
    metrics = sim.run(trades)
    logger.info(metrics.summary())
    return metrics
