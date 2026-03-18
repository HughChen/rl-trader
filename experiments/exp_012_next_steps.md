# Experiment 012: Next Steps (Stronger Constraints + Curriculum + Regime)

## Setup

Implements the suggested next steps from exp_011:

| Change | Value |
|--------|-------|
| **Stronger constraints** | min_rebalance_interval=12, max_turnover=0.1 |
| **Curriculum** | Phase 1: 50-step episodes (60k steps), Phase 2: 200-step episodes (60k steps) |
| **Regime-aware** | regime_bias=1.0 (2× weight for bear periods) |

### Command

```bash
python scripts/train_and_evaluate.py --total-timesteps 120000 --cost-analysis --lr-decay --curriculum --regime-bias 1.0
```

---

## Results

### Test

| Metric | Rebalancing | Buy-and-hold | PPO Agent |
|--------|-------------|--------------|-----------|
| Sharpe | -3.33 | -0.44 | -3.28 |
| Total Ret | -81.99% | -29.41% | -99.99% |

**Agent vs rebalancing**: Sharpe +0.05 (agent slightly better)

### Val

| Metric | Rebalancing | Buy-and-hold | PPO Agent |
|--------|-------------|--------------|-----------|
| Sharpe | -3.50 | -0.47 | -3.71 |
| Total Ret | -87.65% | -33.58% | -100% |

### Cost attribution (test, agent)

| Config | Total Return | Cum Fee | Cum Slippage |
|--------|--------------|---------|--------------|
| Frictionless | -36.66% | — | — |
| Fees only | -36.88% | 0.35% | — |
| Full costs | -79.77% | 0.35% | 113.75% |

---

## Observations

1. **Agent beats rebalancing** on test Sharpe (+0.05) — first time.
2. **Turnover greatly reduced**: cum fee 0.35% (vs 3.6% exp_011, 545% exp_007).
3. **Slippage reduced**: 114% cum (vs 211% exp_011).
4. **Agent still blows up** (-99.99% return) — worse than buy-and-hold (-29%).
5. **Val performance worse** than rebalancing (-0.22 Sharpe diff).

---

## Conclusion

Stronger constraints + curriculum + regime bias reduce churning and let the agent beat rebalancing on test Sharpe. Return is still -100%, so the policy is not practically useful. Buy-and-hold remains the best baseline.
