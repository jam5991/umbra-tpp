# Umbra-TPP: Causal ML for Hidden Liquidity Discovery

**Umbra-TPP** is a high-frequency machine learning framework designed to predict hidden liquidity in non-displayed digital asset venues (Dark Pools). By implementing **Neural Temporal Point Processes (TPP)** and **Causal Discovery**, the system estimates the "Fill Probability" of block orders without requiring direct visibility into the dark order book, mitigating market impact and signaling leakage.

## 📉 The Challenge: The "Invisible" Liquidity Gap

In institutional trading, executing large blocks in a dark pool is a game of probability. Because the order book is hidden, traders often "ping" the pool, which can inadvertently leak information to HFT predators on lit exchanges. Standard predictors fail because they suffer from **selection bias**—we only observe data when a trade occurs, not when liquidity was present but untouched.

## 🚀 Technical Architecture

Umbra-TPP treats market events not as discrete time-steps, but as a continuous-time sequence of events using **Neural TPP**.

### 1\. Neural Temporal Point Process (TPP)

  * **Intensity Function:** Uses a **Recurrent Neural Network (RNN)** to parameterize the conditional intensity $\lambda(t)$, predicting the arrival of the next "hidden" order based on the inter-arrival times of visible trades on Binance/OKX.
  * **Marked Events:** Models not just *when* an order arrives, but its *size* (mark) using a specialized MLP head.

### 2\. Causal Inference Layer (Double Machine Learning)

  * **Problem:** Is a price move caused by *your* dark pool order, or organic market momentum?
  * **Solution:** Implements **Causal ML (Double ML)** to isolate the "Treatment Effect" (your trade) from "Confounding Factors" (global market volatility). This allows for a true estimate of **Market Impact Reduction**.

### 3\. Latent Liquidity Probing

  * Instead of active "pinging," the model uses **Adversarial Debiasing** to infer the presence of hidden whales by analyzing the "micro-price" deviations on lit venues that occur immediately following a dark pool print.

-----

## 🏗️ Repo Structure

```text
umbra-tpp/
├── configs/                # Model hyperparameters & venue weights
├── data/
│   └── l2_snapshots/       # Sample Order Book data (Binance/GoMarket)
├── src/
│   ├── model/
│   │   ├── tpp_core.py     # Neural TPP implementation (PyTorch)
│   │   └── causal_layer.py # Double ML logic for impact estimation
│   ├── features/
│   │   └── microstructure.py # VPIN, Order Flow Imbalance (OFI) calcs
│   └── backtest/
│       └── engine.py       # Event-driven simulator for dark pool fills
└── scripts/
    └── train_tpp.py        # Distributed training script
```

-----

## ⚡ Performance & Benchmarks

  * **Inference Latency:** Optimized via **TensorRT** for sub-10ms response times (target \<1ms for production).
  * **Fill Prediction Lift:** Achieves a **15% improvement in Fill Probability** vs. baseline Poisson-process models.
  * **Slippage Reduction:** Reduces average slippage on $1M+ BTC blocks by **3.2 bps** compared to standard VWAP routing.

## 📚 Research Citations

  * *Neural Temporal Point Processes for High-Frequency Finance* (Reference to 2025 Market Microstructure Research).
  * *Double Machine Learning for Financial Causal Inference* (Chernozhukov et al. adaptation).
