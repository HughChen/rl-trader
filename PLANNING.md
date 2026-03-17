# RL-Based Cryptocurrency Trader — Planning Document

> Offline backtesting and methodology evaluation for PGPortfolio-style reinforcement learning trading strategies.

**See also**: [experiments/](experiments/) for setup details and findings.

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

### 4.1 Exchange Restrictions (CCXT)

| Exchange | History limit | Notes |
|----------|---------------|-------|
| **KuCoin** | ~2 years (1h bars) | ✅ Recommended. Paginates correctly, no geo-restriction. |
| **Kraken** | ~720 candles max | ❌ Ignores `since` beyond limit. 1h bars ≈ 1 month only. |
| **Binance** | Years | ⚠️ Geo-restricted (451) in US and some regions. |
| **Bybit** | Varies | Untested; may work similarly to KuCoin. |

**Recommendation**: Use KuCoin (`--exchange kucoin`) for 2+ years of data. Kraken is unsuitable for RL (too little history). Binance works only if not geo-restricted.

### 4.2 Phase 1: Easy Path (Start Here)

- **Fetcher**: CCXT — pull 1h OHLCV for top 10 liquid coins (BTC, ETH, SOL, etc.)
- **History**: 2+ years for sufficient RL training samples (use KuCoin)
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
- [x] Gymnasium-compatible trading environment
- [x] Observation space: price tensor + indicators + portfolio state
- [x] Action space: portfolio weights (continuous)
- [x] Reward: log-return with transaction cost penalty
- [x] Fee model (configurable maker/taker)
- [x] Slippage model (square-root market impact using volume)

---

## 12. Phase 2: Environment — Implementation Plan

### 12.1 Overview

Build `src/env/` — a Gymnasium environment that steps through aligned market data, accepts portfolio weight actions, and returns observations + rewards with fees and slippage.

### 12.2 Data Loading & Alignment

| Task | Details |
|------|---------|
| **Input** | `load_splits()` returns train/val/test dicts of symbol → DataFrame |
| **Alignment** | Intersect timestamps across all symbols; drop rows with any NaN in required columns |
| **Warmup** | Indicators have warmup (e.g. RSI needs 14 bars). Start env step index after warmup, or forward-fill/drop early rows |
| **Output** | Single aligned DataFrame with MultiIndex (timestamp, symbol) or dict of aligned arrays indexed by step |

**Structure**: Create `AlignedMarketData` helper that:
- Takes `dict[str, DataFrame]`, aligns on common timestamps
- Exposes `prices[t, i]`, `volumes[t, i]`, `indicators[t, i, :]` by step index
- Handles warmup by trimming first N rows

### 12.3 Observation Space

**Components** (flattened into a 1D vector for Gymnasium `Box`):

| Component | Shape | Description |
|-----------|-------|--------------|
| **Price tensor** | `(history_window × n_assets)` | Last `history_window` steps of normalized close returns: `r_t = close_t / close_{t-1} - 1` (or log return). Per-asset, last H values. |
| **Indicators** | `(n_assets × n_indicators)` | RSI, MACD, MACD_signal, MACD_diff, BB position, ATR (normalized), rel_volume. One value per asset per indicator. |
| **Portfolio state** | `(n_assets,)` | Previous weights w_{t-1} |

**Total dim**: `history_window * n_assets + n_assets * n_indicators + n_assets`

**Normalization**: Clip/mask NaNs; normalize indicators to [0,1] or z-score using rolling stats (optional for v1).

**Config**: `history_window` (e.g. 50), `indicator_cols` (subset of available).

### 12.4 Action Space

- **Type**: `Box(low=0, high=1, shape=(n_assets,))`
- **Constraint**: Weights must sum to 1. Two options:
  1. **Env normalizes**: Clip to [0,1], divide by sum. Agent can output raw logits.
  2. **Policy uses softmax**: Agent outputs logits; softmax gives valid weights. Env receives already-normalized weights.
- **Recommendation**: Env normalizes — more robust to policy errors.

### 12.5 Step Logic (Pseudocode)

```
def step(action):
    w_target = normalize(action)  # ensure sum=1, in [0,1]
    w_prev = self.w
    prices_now = self.prices[t]
    prices_prev = self.prices[t-1]
    volumes_now = self.volumes[t]

    # Portfolio value before rebalancing (price move)
    p_before = sum(w_prev[i] * (prices_now[i] / prices_prev[i]) for i in assets)
    p_prev = 1.0  # or track actual

    # Transaction cost: fee + slippage
    fee_cost = fee_rate * sum(|w_target[i] - w_prev[i]|)
    slippage_cost = sum(slippage_i for each asset i)
    # slippage_i = sigma * sqrt(trade_value_i / daily_volume_i)

    cost_factor = 1 - fee_cost - slippage_cost
    p_after = p_before * cost_factor

    reward = log(p_after / p_prev)
    self.w = w_target
    self.portfolio_value *= (p_after / p_prev)
    t += 1
    obs = build_observation(t)
    done = t >= T
    return obs, reward, done, truncated, info
```

### 12.6 Fee Model

- **Config**: `fee_rate: float` (e.g. 0.001 for 0.1%)
- **Formula**: `cost = fee_rate * sum_i |w_target[i] - w_prev[i]|` (fraction of portfolio traded)
- **Note**: Applied once per rebalance; symmetric for buys/sells.

### 12.7 Slippage Model (Square-Root)

- **Config**: `slippage_sigma: float` (e.g. 0.1)
- **Per asset i**: `trade_value_i = |Δw_i| * portfolio_value`
- **Slippage cost (fraction)**: `sigma * sqrt(trade_value_i / (volume_i * price_i))` — volume in quote currency
- **Total**: Sum over assets. Cap at some max to avoid extreme values.

**Simpler variant**: `slippage = sigma * sqrt(sum(|Δw_i|))` — no volume, just trade size.

### 12.8 Episode Structure

- **Option A**: One episode = full dataset (train/val/test). `reset()` starts at step 0; `done` when reaching end.
- **Option B**: Random sub-episodes for training (sample random start, fixed length). Better for RL sample diversity.
- **Recommendation**: Support both. Param `episode_length=None` for full, or `episode_length=252` for random windows.

### 12.9 File Structure

```
src/env/
├── __init__.py
├── aligned_data.py    # AlignedMarketData: load, align, index by step
├── cost_models.py    # fee_cost(), slippage_cost()
└── portfolio_env.py  # PortfolioEnv(gymnasium.Env)
```

### 12.10 Implementation Order

1. **aligned_data.py** — Load splits, align timestamps, build step-indexed arrays. Unit test on small data.
2. **cost_models.py** — Fee and slippage functions. Unit test with known inputs.
3. **portfolio_env.py** — Env class. `reset()` loads data, sets t=0, returns obs. `step()` implements logic above. Test with random agent.
4. **Integration** — Run a few episodes with equal-weight baseline, verify reward sign and magnitude.

### Phase 3: Agent Training
- [x] Baseline: Equal-weight, buy-and-hold
- [x] RL agent (PPO via Stable-Baselines3)
- [ ] Hyperparameter tuning on validation set

### Phase 4: Evaluation
- [x] Sharpe ratio, Sortino ratio, max drawdown
- [x] Comparison vs. equal-weight portfolio
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

## 13. Improvement Plans

Planned modifications to improve agent robustness and evaluation rigor.

### 13.1 Walk-Forward Testing

**Current**: Single train (2024) → val (early 2025) → test (mid 2025–2026).

**Plan**: Rolling windows to test across multiple regimes:

| Window | Train | Test |
|--------|-------|------|
| 1 | 2024-Q1–Q2 | 2024-Q3 |
| 2 | 2024-Q1–Q3 | 2024-Q4 |
| 3 | 2024-Q2–Q4 | 2025-Q1 |
| … | … | … |

**Implementation**: `scripts/walk_forward_eval.py` — define overlapping windows, train (or load) per window, evaluate on out-of-sample period, report mean/median Sharpe and consistency.

**Benefit**: More robust performance estimate; reduces luck from a single favorable test window.

### 13.2 Action Constraints (Concentration Limits)

**Problem**: Agent can allocate 100% to one asset and blow up.

**Plan**: Cap max weight per asset in env normalization:

```
w = clip(w, 0, max_weight_per_asset)  # e.g. 0.35
w = w / w.sum()
```

**Benefit**: Limits concentration risk and extreme drawdowns.

### 13.3 Risk-Adjusted Reward (Sharpe-Like)

**Current**: Reward = log return (minus costs).

**Plan**: Use Sharpe-like reward so agent optimizes risk-adjusted return:

```
reward = mean_return / (std_return + eps)
```

Or rolling Sharpe over last N steps.

**Benefit**: Encourages stable strategies over high-variance bets.

### 13.4 Regime-Aware Training

**Problem**: Train may be mostly bull (2024), test bear (2025–2026) — regime mismatch.

**Plan**:
- Sample episodes uniformly over time so bear periods aren’t underrepresented
- Or oversample bearish periods during training
- Or add regime labels (bull/bear/sideways) if available

### 13.5 Shorter Episodes

**Current**: 252 steps (~10.5 days of hourly data) per episode.

**Plan**: Try 50–100 steps.

**Benefit**: More episodes per run, faster feedback, potentially better credit assignment.

### 13.6 Implementation Priority

1. **Action constraints** — Quick win, prevents blow-ups
2. **Walk-forward evaluation** — Better assessment of robustness
3. **Risk-adjusted reward** — May improve strategy quality
4. **Shorter episodes** — Easy to test
5. **Regime-aware training** — More involved, do last

---

## References

- Jiang et al., "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" (PGPortfolio)
- FinRL-Meta: https://github.com/AI4Finance-Foundation/FinRL-Meta
- CCXT: https://github.com/ccxt/ccxt
