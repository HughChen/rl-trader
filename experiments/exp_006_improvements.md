# Experiment 006: Robustness Improvements

## Setup

Implements PLANNING.md §13 improvements:

| Change | Implementation |
|-------|----------------|
| **Action constraints** | `max_weight_per_asset=0.35` (default) in env normalization |
| **Shorter episodes** | Default `episode_length=100` (was 252) |
| **Walk-forward eval** | `scripts/walk_forward_eval.py` — 4 rolling windows |
| **Risk-adjusted reward** | Optional `--reward-type sharpe` with rolling window |

### Commands

```bash
# Train with new defaults (shorter episodes, concentration limit)
python scripts/train_agent.py --total-timesteps 200000

# Walk-forward evaluation
python scripts/walk_forward_eval.py --timesteps-per-window 100000

# Try Sharpe reward (experimental)
python scripts/train_agent.py --reward-type sharpe --sharpe-window 20
```

---

## Findings

### Single Split (train 200k steps, eval on test)

| Metric | Baseline | PPO Agent |
|--------|----------|-----------|
| Sharpe | -0.44 | -12.79 |
| Total Return | -32.75% | -100% |
| Max DD | -64.46% | -100% |

**Observation**: Agent still collapses to -100% despite action constraints (max 35% per asset). Concentration limit did not prevent blow-up; likely causes: heavy churning (fees/slippage), or repeatedly picking worst-performing assets.

### Walk-Forward (4 windows, 50k steps/window)

| Window | Train | Test Period | Agent Sharpe | Agent Ret | Baseline Sharpe | Baseline Ret |
|--------|-------|-------------|--------------|-----------|----------------|--------------|
| 1 | ≤ 2024-06-30 | Q3 2024 | -12.87 | -99.96% | 0.25 | -0.44% |
| 2 | ≤ 2024-09-30 | Q4 2024 | -9.03 | -99.94% | 3.08 | +92.31% |
| 3 | ≤ 2024-12-31 | Q1 2025 | -11.38 | -99.99% | -1.31 | -35.52% |
| 4 | ≤ 2025-03-31 | Q2 2025 | -12.98 | -99.97% | 0.76 | +9.79% |

**Summary**: Agent beats baseline 0/4 windows. Mean agent Sharpe: -11.57; mean baseline Sharpe: +0.70. Agent consistently blows up across all regimes (bull Q4 2024, bear Q1 2025).

### Baseline by Split (full period)

| Split | Sharpe | Total Return |
|-------|--------|--------------|
| Train | 0.72 | +33% |
| Val | -0.40 | -29% |
| Test | -0.44 | -33% |

### Cost Attribution (see exp_007)

Agent would be **+57% frictionless**; blow-up is from costs. Slippage dominates (99.9% impact); fees contribute ~39%. Large rebalances cause extreme slippage.

### Conclusions

1. **Action constraints insufficient**: max_weight=0.35 did not prevent -100% collapse.
2. **Walk-forward confirms failure**: Agent fails across multiple train/test windows.
3. **Costs are the killer**: Frictionless +57%; slippage from churning destroys value. See [exp_007](exp_007_cost_attribution.md).
4. **Next directions**: Higher turnover penalty, PGPortfolio-style input, or position-size limits.
