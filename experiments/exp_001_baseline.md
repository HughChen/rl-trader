# Experiment 001: Equal-Weight Baseline

## Setup

| Parameter | Value |
|-----------|-------|
| **Strategy** | Equal-weight buy-and-hold |
| **Data** | KuCoin, 10 symbols, 1h bars, ~2yr |
| **Splits** | Train: Mar–Dec 2024; Val: Jan–Jun 2025; Test: Jul 2025–Mar 2026 |
| **Command** | `python scripts/run_baseline.py --split {train,val,test}` |

No training; strategy holds fixed weights (1/n per asset) throughout.

---

## Findings

| Split | Sharpe | Sortino | Max DD | Total Return |
|-------|--------|---------|--------|--------------|
| Train | 0.72 | 0.88 | -48% | +33% |
| Test | -0.44 | -0.57 | -64% | -33% |

**Observation**: Test period is bearish. Negative baseline reflects market decline, not strategy failure. Beating baseline on test means losing less than -33%.
