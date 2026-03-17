# Experiment 008: Slippage Notional Fix

## Setup

Fixed units bug in slippage model: trade value was in portfolio fractions while volume was in USD. Added `notional_usd` (default 1e6) so:

```
trade_value_usd = |Δw| × portfolio_value × notional_usd
impact = σ × √(trade_value_usd / volume_quote)
```

| Parameter | Value |
|-----------|-------|
| **notional_usd** | 1e6 (configurable via `--notional`) |
| **slippage_sigma** | 0.05 (train), 0.1 (eval) |

---

## Findings

### Before fix (old model, wrong units)

- Frictionless: +57%
- Slippage only: -99.8%, cum_slip=49%
- Agent trained with effectively negligible slippage

### After fix (old model, correct units)

- Slippage only: -100%, cum_slip=1117%
- Slippage now correctly scales with trade size vs volume

### After fix (retrained 200k steps)

| Config | Return |
|--------|--------|
| Frictionless | +19.9% |
| Fees only | -99.9% |
| Slippage only | -100% |
| Full costs | -100% |

**Observation**: Retrained agent has lower frictionless return (+20% vs +57%) — learned to trade less. Still churns too much; costs destroy value. Higher turnover penalty or smaller notional may help.

### Baseline (unchanged)

- Test Sharpe: -0.44, Return: -32.75%

---

## Conclusion

Slippage model now correctly scales with portfolio size. Agent still blows up; needs stronger turnover penalty or other regularization.
