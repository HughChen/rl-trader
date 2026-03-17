# RL-Based Cryptocurrency Trader — Planning Document

> Offline backtesting and methodology evaluation for PGPortfolio-style reinforcement learning trading strategies.

---

## 1. Project Overview

**Goal**: Implement and evaluate RL-based portfolio allocation strategies on historical cryptocurrency data, without production deployment.

**Scope**: Backtesting, methodology comparison, and research validation. Focus on understanding what works before considering live trading.

**Decisions**:
- **Multi-asset**: Portfolio allocation across multiple cryptocurrencies (e.g., top 10 liquid coins)
- **Spot only**: No perpetual futures (no funding rates)
- **Slippage**: Include slippage simulation in the backtest (see §3.4)
- **No LLM/sentiment**: Out of scope for now

---

## 2. Data Frequency Recommendation

| Frequency | Interval | Use Case | Verdict |
|-----------|----------|----------|---------|
| High | 1m–5m | Execution algorithms, HFT | ❌ Fees/slippage eat profits; low signal-to-noise |
| **Intermediate** | **15m–1h** | **PGPortfolio-style portfolio RL** | ✅ **Recommended** |
| Low | 4h–1d | Trend following | ⚠️ Fewer samples; slow reaction |

**Recommendation**: **30-minute bars** — the standard for PGPortfolio research. Balances:
- Enough signal for CNN/RNN pattern recognition
- Filters out micro-structure noise
- Reasonable data volume for training

---

## 3. Data Requirements

### 3.1 Market Feature Data (Observation Space)

What the agent observes at each step:

| Component | Required | Source |
|-----------|----------|--------|
| **OHLCV** | Yes | Open, High, Low, Close, Volume per asset |
| **Price Tensor** | Yes | PGPortfolio-style normalized price history |
| **Technical Indicators** | Yes | RSI, MACD, Bollinger Bands (momentum + volatility) |
| **Volatility (ATR)** | Yes | Average True Range — critical for crypto |
| **Relative Volume** | Recommended | Market "crowdedness" signal |

### 3.2 Portfolio State Data (Context)

| Component | Purpose |
|-----------|---------|
| Previous weights (w_{t-1}) | Compute rebalancing costs |
| Cash / position sizes | Portfolio constraint enforcement |

### 3.3 Environment & Constraint Data (Ground Truth)

| Component | Purpose |
|-----------|---------|
| Fee schedule | Maker/taker fees (e.g., 0.1% Binance) |
| Funding rates | N/A (spot only) |
| Risk-free rate | Sharpe/Sortino benchmark |

### 3.4 Slippage Simulation

We want realistic execution costs beyond fixed fees. Options (from simple to complex):

| Approach | Data Needed | Realism | Effort |
|----------|-------------|---------|--------|
| **Linear in trade size** | None (OHLCV only) | Low | Easy — `slippage = k × |Δw| × volume` |
| **Square-root (Almgren-Chriss)** | Volume | Medium | Easy — market impact ∝ √(trade size / volume) |
| **Order book snapshot** | L2 order book | High | Medium — requires historical depth |
| **Tick-level replay** | Tardis/CoinAPI | Highest | Hard — expensive, large data |

**Recommendation**: Start with **square-root market impact** — no extra data beyond OHLCV volume, captures the intuition that larger trades move price more. Formula: `impact ≈ σ × √(trade_value / daily_volume)` where σ is a configurable impact coefficient.

---

## 4. Data Acquisition Strategy

### Phase 1: Easy Path (Start Here)

- **Fetcher**: CCXT — pull 1h OHLCV for top 10 liquid coins (BTC, ETH, SOL, etc.)
- **History**: 2+ years for sufficient RL training samples
- **Feature engineering**: ta library for ATR, RSI, MACD
- **Cost**: Free / low cost

### Phase 2: Medium Path (If Needed)

- **FinRL-Meta**: Pre-built data processors, RL environments
- **altFINS**: Pre-calculated technical indicators (12 months, hobbyist tier)

### Phase 3: Hard Path (Optional — Higher-Fidelity Slippage)

- **Tardis.dev / CoinAPI**: Tick-level, order book data for realistic slippage
- **Cost**: $100–500/month — consider only if volume-based slippage proves insufficient

---

## 5. Reward Function Design

**Critical**: Penalize trading to avoid churning.

PGPortfolio-style log-return with cost factor:

```
r_t = ln(μ_t · p_t / p_{t-1})
```

Where:
- `μ_t` = cost factor (1 - fees - slippage) — both fees and slippage reduce the realized return
- `p_t` = portfolio value at time t

**Implementation tips**:
- Explicitly model rebalancing cost in the reward
- Consider transaction cost as a function of `|w_t - w_{t-1}|`

---

## 6. Data Augmentation

RL is data-hungry. Strategies:

1. **Sliding window**: Create overlapping training samples from 1h bars
2. **Temporal augmentation**: Slight time-shifts (if applicable)
3. **Asset universe**: Multi-asset portfolio across multiple coins — increases diversity and sample efficiency

---

## 7. Train–Test–Trade Split (Walk-Forward Validation)

| Split | Period | Purpose |
|-------|--------|---------|
| **In-Sample (Train)** | 2023–2024 | Model training |
| **Validation** | Early 2025 | Hyperparameter tuning |
| **Out-of-Sample (Test)** | Late 2025–Current | Final evaluation |

**Why**: If the agent only profits in a bull market, it hasn't learned a strategy — it's lucky.

---

## 8. Suggested Tech Stack

| Layer | Tool |
|-------|------|
| Data fetching | CCXT |
| Feature engineering | ta (Technical Analysis Library) |
| RL environment | FinRL (Portfolio Allocation), or custom Gymnasium env |
| RL algorithms | Stable-Baselines3, or custom PG implementation |
| Backtesting | Custom env with fee/slippage simulation |

---

## 9. Implementation Phases

### Phase 1: Data Pipeline
- [ ] CCXT fetcher for OHLCV (1h bars, top 10 coins, 2 years)
- [ ] Technical indicator computation (ATR, RSI, MACD)
- [ ] CSV/Parquet storage with reproducible schema
- [ ] Train/validation/test split by date

### Phase 2: Environment
- [ ] Gymnasium-compatible trading environment
- [ ] Observation space: price tensor + indicators + portfolio state
- [ ] Action space: portfolio weights (continuous or discrete)
- [ ] Reward: log-return with transaction cost penalty
- [ ] Fee model (configurable maker/taker)
- [ ] Slippage model (square-root market impact using volume)

### Phase 3: Agent Training
- [ ] Baseline: Equal-weight, buy-and-hold
- [ ] RL agent (e.g., PPO, A2C, or PGPortfolio-style policy gradient)
- [ ] Hyperparameter tuning on validation set

### Phase 4: Evaluation
- [ ] Sharpe ratio, Sortino ratio, max drawdown
- [ ] Comparison vs. BTC benchmark, equal-weight portfolio
- [ ] Sensitivity analysis: fee levels, slippage coefficient, bar frequency
- [ ] Ablation: with/without technical indicators, with/without cost penalty

---

## 10. Success Criteria

- Agent beats equal-weight on out-of-sample Sharpe (after fees)
- Robust across train/validation/test periods (not just bull market)
- Transaction cost sensitivity: strategy degrades gracefully as fees and slippage increase
- Reproducible: fixed seeds, versioned data, documented pipeline

---

## 11. Open Questions / Future Work

- [ ] Upgrade to order book–based slippage if volume model is too coarse?
- [ ] Add perpetual futures support later (funding rates) if shorting/leverage desired?

---

## References

- Jiang et al., "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" (PGPortfolio)
- FinRL-Meta: https://github.com/AI4Finance-Foundation/FinRL-Meta
- CCXT: https://github.com/ccxt/ccxt
